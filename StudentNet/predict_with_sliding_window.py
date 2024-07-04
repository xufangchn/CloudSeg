import math
import torch

class Predict():
    def _predict_with_sliding_window(self, optical_data, SAR_data):

        height, width = optical_data.shape[2:]
        step_size = self.opts.model_train_size // 2
        self.width_step_num = math.ceil(width / step_size) - 1
        self.height_step_num = math.ceil(height / step_size) - 1

        n_evals = self.width_step_num * self.height_step_num

        self.pred_total = torch.zeros([self.height_step_num, self.width_step_num, self.model.num_classes, self.opts.output_patch_size, self.opts.output_patch_size]).cuda()

        for index in range(n_evals):
            h_index, w_index = index // self.width_step_num, index % self.width_step_num

            y, x = h_index * step_size, w_index * step_size
            y_start, y_valid = self._cal_start_valid(y, self.opts.model_train_size, height)
            x_start, x_valid = self._cal_start_valid(x, self.opts.model_train_size, width)

            optical_data_p = optical_data[..., y_start:y_start + self.opts.model_train_size, x_start:x_start + self.opts.model_train_size]
            SAR_data_p = SAR_data[..., y_start:y_start + self.opts.model_train_size, x_start:x_start + self.opts.model_train_size]
            pred_p = self.model.forward(optical_data=optical_data_p,
                                        SAR_data=SAR_data_p,
                                        output_shape=[self.opts.output_patch_size, self.opts.output_patch_size])

            _y_valid = self._rescale_value(y_valid)
            _x_valid = self._rescale_value(x_valid)

            self.pred_total[h_index, w_index, :, :_y_valid, :_x_valid] = pred_p[0, :, self.opts.output_patch_size - _y_valid:, self.opts.output_patch_size - _x_valid:]

        return self._merge_predictions(height, width, step_size)

    def _merge_predictions(self, height, width, step_size):
        self._height = self._rescale_value(height)
        self._width = self._rescale_value(width)
        self._step_size = self._rescale_value(step_size)

        pred_merge = torch.zeros([1, self.model.num_classes, self._height, self._width]).cuda()

        for h in range(self.height_step_num + 1):
            for w in range(self.width_step_num + 1):
                merge_y_start = h * step_size
                merge_y_end = min(merge_y_start + step_size, height)
                merge_x_start = w * step_size
                merge_x_end = min(merge_x_start + step_size, width)

                self._merge_y_start = self._rescale_value(merge_y_start)
                self._merge_y_end = self._rescale_value(merge_y_end)
                self._merge_x_start = self._rescale_value(merge_x_start)
                self._merge_x_end = self._rescale_value(merge_x_end)
                self.get_boundary()

                pred_merge[:, :, self._merge_y_start:self._merge_y_end, self._merge_x_start:self._merge_x_end] = self.assign_merge_value(h, w)

        return pred_merge

    def assign_merge_value(self, h, w):
        if h == 0:
            if w == 0:
                return self.get_bottom_right(h, w)
            elif w == self.width_step_num:
                return self.get_bottom_left(h, w)
            else:
                return (self.get_bottom_left(h, w) + self.get_bottom_right(h, w)) / 2
        elif h == self.height_step_num:
            if w == 0:
                return self.get_top_right(h, w)
            elif w == self.width_step_num:
                return self.get_top_left(h, w)
            else:
                return (self.get_top_left(h, w) + self.get_top_right(h, w)) / 2
        else:
            if w == 0:
                return (self.get_top_right(h, w) + self.get_bottom_right(h, w)) / 2
            elif w == self.width_step_num:
                return (self.get_top_left(h, w) + self.get_bottom_left(h, w)) / 2
            else:
                return (self.get_top_left(h, w) + self.get_top_right(h, w) + self.get_bottom_left(h, w) + self.get_bottom_right(h, w)) / 4

    def get_boundary(self):
        self.h_b = min(self._step_size + self._height - self._merge_y_start, self.opts.output_patch_size)
        self.w_b = min(self._step_size + self._width - self._merge_x_start, self.opts.output_patch_size)

    def get_top_left(self, h, w):
        return self.pred_total[h - 1, w - 1, :, self._step_size: self.h_b, self._step_size: self.w_b] 

    def get_top_right(self, h, w):
        return self.pred_total[h - 1, w, :, self._step_size: self.h_b, :self._step_size]
		
    def get_bottom_left(self, h, w):
        return self.pred_total[h, w - 1, :, :self._step_size, self._step_size: self.w_b]

    def get_bottom_right(self, h, w):
        return self.pred_total[h, w, :, :self._step_size, :self._step_size]

    def _cal_start_valid(self, fst, window_size, boundary):
        if fst + window_size > boundary:
            start = boundary - window_size
            valid = boundary - fst
        else:
            start = fst
            valid = window_size
        return start, valid

    def _rescale_value(self, value):
        _value = value / self.opts.model_train_size * self.opts.output_patch_size
        assert _value.is_integer
        _value = int(_value)
        return _value
