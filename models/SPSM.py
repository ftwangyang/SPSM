import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
__all__ = ['SPSM']


class SPSM(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.eval_type = self.model_params['eval_type']
        self.problem = self.model_params['problem']

        self.encoder = MTL_Encoder(**model_params)
        self.decoder = MTL_Decoder(**model_params)
        self.encoded_nodes = None  # shape: (batch, problem+1, EMBEDDING_DIM)
        self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in model_params.keys() else model_params['device']
        # assert self.model_params['norm'] == "batch", "The original MTLModel implementation uses batch normalization."

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        node_tw_start = reset_state.node_tw_start
        node_tw_end = reset_state.node_tw_end
        # shape: (batch, problem)
        node_xy_demand_tw = torch.cat((node_xy, node_demand[:, :, None], node_tw_start[:, :, None], node_tw_end[:, :, None]), dim=2)
        # shape: (batch, problem, 5)

        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand_tw)
        # shape: (batch, problem+1, embedding)
        self.decoder.set_kv(self.encoded_nodes)

    def set_eval_type(self, eval_type):
        self.eval_type = eval_type

    def forward(self, state, selected=None):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long).to(self.device)
            prob = torch.ones(size=(batch_size, pomo_size))
            # probs = torch.ones(size=(batch_size, pomo_size, self.encoded_nodes.size(1)))
            # shape: (batch, pomo, problem_size+1)

            # # Use Averaged encoded nodes for decoder input_1
            # encoded_nodes_mean = self.encoded_nodes.mean(dim=1, keepdim=True)
            # # shape: (batch, 1, embedding)
            # self.decoder.set_q1(encoded_nodes_mean)

            # # Use encoded_depot for decoder input_2
            # encoded_first_node = self.encoded_nodes[:, [0], :]
            # # shape: (batch, 1, embedding)
            # self.decoder.set_q2(encoded_first_node)

        elif state.selected_count == 1:  # Second Move, POMO
            # selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, -1).to(self.device)
            selected = state.START_NODE
            prob = torch.ones(size=(batch_size, pomo_size))
            # probs = torch.ones(size=(batch_size, pomo_size, self.encoded_nodes.size(1)))

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            attr = torch.cat((state.load[:, :, None], state.current_time[:, :, None], state.length[:, :, None], state.open[:, :, None]), dim=2)
            # shape: (batch, pomo, 4)
            probs = self.decoder(encoded_last_node, attr, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem+1)
            if selected is None:
                while True:
                    if self.training or self.eval_type == 'softmax':
                        try:
                            selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)
                        except Exception as exception:
                            print(">> Catch Exception: {}, on the instances of {}".format(exception, state.PROBLEM))
                            exit(0)
                    else:
                        selected = probs.argmax(dim=2)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)

                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break
            else:
                selected = selected
                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class MTL_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(5, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand_tw):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand_tw.shape: (batch, problem, 5)

        embedded_depot = self.embedding_depot(depot_xy)
        # shape: (batch, 1, embedding)
        embedded_node = self.embedding_node(node_xy_demand_tw)
        # shape: (batch, problem, embedding)

        out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape: (batch, problem+1, embedding)

        for layer in self.layers:
            out = layer(out)

        return out
        # shape: (batch, problem+1, embedding)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        self.gate = nn.Linear(embedding_dim, embedding_dim)

        # Shared attention layer for common features
        self.Wq_shared = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk_shared = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_shared = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        # Task-specific attention layer for task-specific features
        self.Wq_task = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk_task = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_task = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine_shared = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.multi_head_combine_task = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = FeedForward(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        head_num = self.model_params['head_num']

        # Shared attention
        q_shared = reshape_by_heads(self.Wq_shared(input1), head_num=head_num)
        k_shared = reshape_by_heads(self.Wk_shared(input1), head_num=head_num)
        v_shared = reshape_by_heads(self.Wv_shared(input1), head_num=head_num)
        shared_out_concat = multi_head_attention(q_shared, k_shared, v_shared)
        shared_out = self.multi_head_combine_shared(shared_out_concat)

        # Task-specific attention
        q_task = reshape_by_heads(self.Wq_task(input1), head_num=head_num)
        k_task = reshape_by_heads(self.Wk_task(input1), head_num=head_num)
        v_task = reshape_by_heads(self.Wv_task(input1), head_num=head_num)
        task_out_concat = multi_head_attention(q_task, k_task, v_task)
        task_out = self.multi_head_combine_task(task_out_concat)

        # Gate mechanism to control the flow of information between shared and task-specific layers
        gate = torch.sigmoid(self.gate(input1))

        gated_out = gate * shared_out + (1 - gate) * task_out

        if self.model_params['norm_loc'] == "norm_last":
            out1 = self.addAndNormalization1(input1, gated_out)
            out2 = self.feedForward(out1)
            out3 = self.addAndNormalization2(out1, out2)
        else:
            out1 = self.addAndNormalization1(None, input1)
            gated_out = self.multi_head_combine_task(out1)
            input2 = input1 + gated_out
            out2 = self.addAndNormalization2(None, input2)
            out2 = self.feedForward(out2)
            out3 = input2 + out2

        return out3


########################################
# DECODER
########################################

class MTL_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        self.gate = nn.Linear(embedding_dim + 4, embedding_dim)

        # Shared attention layer for common features
        self.Wq_shared = nn.Linear(embedding_dim + 4, head_num * qkv_dim, bias=False)
        self.Wk_shared = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_shared = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        # Task-specific attention layer for task-specific features
        self.Wq_task = nn.Linear(embedding_dim + 4, head_num * qkv_dim, bias=False)
        self.Wk_task = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_task = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine_shared = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.multi_head_combine_task = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k_shared = None
        self.v_shared = None
        self.k_task = None
        self.v_task = None
        self.single_head_key_shared = None
       # self.single_head_key_task = None

    def set_kv(self, encoded_nodes):
        head_num = self.model_params['head_num']

        self.k_shared = reshape_by_heads(self.Wk_shared(encoded_nodes), head_num=head_num)
        self.v_shared = reshape_by_heads(self.Wv_shared(encoded_nodes), head_num=head_num)

        self.k_task = reshape_by_heads(self.Wk_task(encoded_nodes), head_num=head_num)
        self.v_task = reshape_by_heads(self.Wv_task(encoded_nodes), head_num=head_num)

        self.single_head_key_shared = encoded_nodes.transpose(1, 2)
        #self.single_head_key_task = encoded_nodes.transpose(1, 2)

    def forward(self, encoded_last_node, attr, ninf_mask):
        head_num = self.model_params['head_num']

        input_cat = torch.cat((encoded_last_node, attr), dim=2)
        q_shared = reshape_by_heads(self.Wq_shared(input_cat), head_num=head_num)
        q_task = reshape_by_heads(self.Wq_task(input_cat), head_num=head_num)

        out_concat_shared = multi_head_attention(q_shared, self.k_shared, self.v_shared, rank3_ninf_mask=ninf_mask)
        out_concat_task = multi_head_attention(q_task, self.k_task, self.v_task, rank3_ninf_mask=ninf_mask)

        shared_out = self.multi_head_combine_shared(out_concat_shared)
        task_out = self.multi_head_combine_task(out_concat_task)

        gate = torch.sigmoid(self.gate(input_cat))
        gated_out = gate * shared_out + (1 - gate) * task_out

        score = torch.matmul(gated_out, self.single_head_key_shared)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)

        return probs



########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.add = True if 'norm_loc' in model_params.keys() and model_params['norm_loc'] == "norm_last" else False
        if model_params["norm"] == "batch":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=True)
        elif model_params["norm"] == "batch_no_track":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "instance":
            self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "layer":
            self.norm = nn.LayerNorm(embedding_dim)
        elif model_params["norm"] == "rezero":
            self.norm = torch.nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        else:
            self.norm = None

    def forward(self, input1=None, input2=None):
        # input.shape: (batch, problem, embedding)
        if isinstance(self.norm, nn.InstanceNorm1d):
            added = input1 + input2 if self.add else input2
            transposed = added.transpose(1, 2)
            # shape: (batch, embedding, problem)
            normalized = self.norm(transposed)
            # shape: (batch, embedding, problem)
            back_trans = normalized.transpose(1, 2)
            # shape: (batch, problem, embedding)
        elif isinstance(self.norm, nn.BatchNorm1d):
            added = input1 + input2 if self.add else input2
            batch, problem, embedding = added.size()
            normalized = self.norm(added.reshape(batch * problem, embedding))
            back_trans = normalized.reshape(batch, problem, embedding)
        elif isinstance(self.norm, nn.LayerNorm):
            added = input1 + input2 if self.add else input2
            back_trans = self.norm(added)
        elif isinstance(self.norm, nn.Parameter):
            back_trans = input1 + self.norm * input2 if self.add else self.norm * input2
        else:
            back_trans = input1 + input2 if self.add else input2

        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
