import dgl
import hgfp
from dgl import data
import torch
import sys
import ast
import gnn_charge
import numpy as np

from matplotlib import pyplot as plt
import time
from time import localtime, strftime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import os

def train(path, config, batch_size=16, learning_rate=1e-5, n_epoches=50):

    time_str = strftime("%Y-%m-%d_%H_%M_%S", localtime())
    os.mkdir(time_str)

    gs_, _ = dgl.data.utils.load_graphs(path)

    gs = []
    for g_ in gs_:
        g = hgfp.heterograph.from_graph(g_)
        g.nodes['atom'].data['q'] = g_.ndata['am1_charge'] + g_.ndata['bcc_charge']
        gs.append(g)

    gs_batched = []

    while True:
        try:
            gs_batched.append(dgl.batch_hetero([gs.pop(0) for _ in range(batch_size)]))
        except:
            break

    ds_tr, ds_te, ds_vl = hgfp.data.utils.split(gs_batched, 1, 1)

    net = gnn_charge.models.Net(config)
    eq = gnn_charge.eq.ChargeEquilibrium()

    opt = torch.optim.Adam(
        list(net.parameters()) + list(eq.parameters()),
        learning_rate)

    loss_fn = torch.nn.functional.mse_loss

    losses = np.array([0.])
    rmse_vl = []
    r2_vl = []
    rmse_tr = []
    r2_tr = []

    for idx_epoch in range(n_epoches):
        for g in ds_tr:
            g_hat = eq(net(g))
            q = g.nodes['atom'].data['q']
            q_hat = g_hat.nodes['atom'].data['q_hat']
            loss = loss_fn(q, q_hat)

            opt.zero_grad()
            loss.backward()
            opt.step()

        net.eval()

        u_tr = np.array([0.])
        u_hat_tr = np.array([0.])

        u_vl = np.array([0.])
        u_hat_vl = np.array([0.])

        with torch.no_grad():
            for g in ds_tr:
                u_hat = eq(net(g)).nodes['atom'].data['q_hat']
                u = g.nodes['atom'].data['q']
                u_tr = np.concatenate([u_tr, u.detach().numpy()], axis=0)
                u_hat_tr = np.concatenate([u_hat_tr, u_hat.detach().numpy()], axis=0)

            for g in ds_vl:
                u_hat = eq(net(g)).nodes['atom'].data['q_hat']
                u = g.nodes['atom'].data['q']
                u_vl = np.concatenate([u_vl, u.detach().numpy()], axis=0)
                u_hat_vl = np.concatenate([u_hat_vl, u_hat.detach().numpy()], axis=0)

        u_tr = u_tr[1:]
        u_vl = u_vl[1:]
        u_hat_tr = u_hat_tr[1:]
        u_hat_vl = u_hat_vl[1:]

        rmse_tr.append(
            np.sqrt(
                mean_squared_error(
                    u_tr,
                    u_hat_tr)))

        rmse_vl.append(
            np.sqrt(
                mean_squared_error(
                    u_vl,
                    u_hat_vl)))

        r2_tr.append(
            r2_score(
                u_tr,
                u_hat_tr))

        r2_vl.append(
            r2_score(
                u_vl,
                u_hat_vl))

        plt.style.use('fivethirtyeight')
        plt.figure()
        plt.plot(rmse_tr[1:], label=r'$RMSE_\mathtt{TRAIN}$')
        plt.plot(rmse_vl[1:], label=r'$RMSE_\mathtt{VALIDATION}$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(time_str + '/RMSE.jpg')
        plt.close()
        plt.figure()
        plt.plot(r2_tr[1:], label=r'$R^2_\mathtt{TRAIN}$')
        plt.plot(r2_vl[1:], label=r'$R^2_\mathtt{VALIDATION}$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(time_str + '/R2.jpg')
        plt.close()
        plt.figure()
        plt.plot(losses[10:])
        plt.title('loss')
        plt.tight_layout()
        plt.savefig(time_str + '/loss.jpg')
        plt.close()





if __name__ == '__main__':
    train(sys.argv[1], ast.literal_eval(sys.argv[2]))
