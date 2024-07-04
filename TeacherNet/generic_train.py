import os
import time
import torch
import numpy as np
from metrics import StreamSegMetrics 

class Generic_Train():
	def __init__(self, model, opts, train_dataloader, val_dataloader):
		self.model=model
		self.opts=opts
		self.train_dataloader=train_dataloader
		self.val_dataloader=val_dataloader

	def train(self):
		
		total_steps = 0
		log_loss = 0
		best_semantic_score = 0 

		for epoch in range(self.opts.max_epochs):
			if epoch < self.model.start_epoch+1:
				pass
				'''
				for data in self.train_dataloader:
					total_steps+=1
					if total_steps % self.opts.log_iter == 0:
						print('epoch', epoch, 'steps', total_steps)
				'''
			else:
				for data in self.train_dataloader:
					total_steps+=1

					self.model.set_input(data)
					batch_loss = self.model.optimize_parameters()
					log_loss = log_loss + batch_loss

					if total_steps % self.opts.log_iter == 0:
						avg_log_loss = log_loss/self.opts.log_iter
						print('epoch', epoch, 'steps', total_steps, 'loss', avg_log_loss)
						log_loss = 0
						

				if (epoch+1) % self.opts.val_freq == 0:
					print("validation...")
					self.model.net_G.eval() 
					with torch.no_grad():
						_iter = 0
						semantic_metric = StreamSegMetrics(self.model.num_classes) 
						for data in self.val_dataloader:
							self.model.set_input(data)
							pred_landcover_data = self.model.forward(optical_data=self.model.cloudfree_data, 
                                                                     SAR_data=self.model.SAR_data, 
                                                                     output_shape=[self.model.output_patch_size, 
																				   self.model.output_patch_size]
																	)
							semantic_metric.update(
								self.model.landcover_data.cpu().numpy(), 
								pred_landcover_data.detach().max(dim=1)[1].cpu().numpy()
							) 
							_iter += 1
						semantic_score = semantic_metric.get_results(semantic_metric.confusion_matrix)["Mean IoU"] 
					print(f'Mean IoU: {semantic_score}') 
					if semantic_score > best_semantic_score:  # save best semantic model
						best_semantic_score = semantic_score
						self.model.save_checkpoint('best_semantic')
					self.model.net_G.train() 
				
				if epoch >= self.opts.lr_start_epoch_decay - self.opts.lr_step:
					self.model.update_lr()
				
				if epoch % self.opts.save_freq == 0:
					self.model.save_checkpoint(epoch)



