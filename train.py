from tqdm.auto import tqdm
import time
import torch

def train_model(num_epochs, model, loss_fn, optimiser, x_train, y_train, hist):
    for t in tqdm(range(num_epochs)):
        log_dir = 'pretrain/'
        model.train()
        start = time.time() 
        y_train_pred = model(x_train)

        loss = loss_fn(y_train_pred, y_train)
        if t % 10 == 0 and t !=0:
            time_ = (time.time() - start)/60
            #print("Epoch ", t, "MSE: ", loss.item())
            my_formatter = "{0:.2f}"
            log_msg = str('time = {}, epoch = {}, loss = {}'.format(my_formatter.format(time_),
                                                                           t + 1,
                                                                           my_formatter.format(loss)))
            with open(f'{log_dir}/train.log', 'a') as f:
                f.write(log_msg + '\n')
            if t % 100 == 0 and t != 0:
                ckpt = model.state_dict()
                torch.save(ckpt, f=f"{log_dir}/LSTM_{t}.pt")
        hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()