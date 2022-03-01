import numpy as np
import utils 


def update_param(dx, learning_rate=1e-2):
    """
    Implementation of standard gradient descent algorithm.
    """
    return learning_rate * dx

def update_param_adagrad(dx, mx, learning_rate=1e-2):
    """
    Implementation of adagrad algorithm.
    """
    return learning_rate * dx / np.sqrt(mx+1e-8)

def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

class Layers():
    def __init__(self):
        """
        store: used to store variables and pass information from forward to backward pass. 
        """
        self.store = None

class FullyConnectedLayer(Layers):
    def __init__(self, dim_in, dim_out):
        """
        Implementation of a fully connected layer.

        dim_in: Number of neurons in previous layer.
        dim_out: Number of neurons in current layer.
        w: Weight matrix of the layer.
        b: Bias vector of the layer.
        dw: Gradient of weight matrix.
        db: Gradient of bias vector
        """
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.w = np.random.uniform(-1, 1, (dim_in, dim_out)) / max(dim_in, dim_out)
        self.b = np.random.uniform(-1, 1, (dim_out,)) / max(dim_in, dim_out)
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        Forward pass of fully connencted layer.

        x: Input to layer (either of form Nxdim_in or in tensor form after convolution NxCxHxW).
        store: Store input to layer for backward pass.
        """
        self.store = x
        #Make sure to flatten input in case input is from a conv or maxpool layer
        reshaped_x = x.reshape(x.shape[0], -1)
        
        out = reshaped_x @ self.w + self.b

        return out

    def backward(self, delta):
        """
        Backward pass of fully connencted layer.

        delta: Error from succeeding layer
        dx: Loss derivitive that that is passed on to layers below
        store: Store input to layer for backward passs
        """
        #Flatten input
        reshaped_x = self.store.reshape(self.store.shape[0], -1)

        dx = delta @ self.w.T
        self.dw = reshaped_x.T @ delta
        self.db = np.sum(delta,axis=0)

        # Upades the weights and bias using the computed gradients
        self.w -= update_param(self.dw)
        self.b -= update_param(self.db)
        return dx.reshape(self.store.shape)


class ConvolutionalLayer(Layers):
    def __init__(self, filtersize, pad=0, stride=1):
        """
        Implementation of a convolutional layer.

        filtersize = (C_out, C_in, F_H, F_W)
        w: Weight tensor of layer.
        b: Bias vector of layer.
        dw: Gradient of weight tensor.
        db: Gradient of bias vector
        """
        self.filtersize = filtersize
        self.pad = pad
        self.stride = stride
        self.w = np.random.normal(0, 0.1, filtersize)
        self.b = np.random.normal(0, 0.1, (filtersize[0],))
        self.dw = None
        self.db = None
    
    def forward(self, x):
        """
        Forward pass of convolutional layer.
        
        x_col: Input tensor reshaped to matrix form.
        store_shape: Save shape of input tensor for backward pass.
        store_col: Save input tensor on matrix from for backward pass.
        """
        N, C, H, W = x.shape
        F, C, HH, WW = self.filtersize 
        
        Wout = int((W - self.filtersize[3]+2*self.pad)/self.stride+1)
        Hout = int((H - self.filtersize[2]+2*self.pad)/self.stride+1)
        
        self.store = (utils.im2col_indices(x, HH, WW, self.pad, self.stride),(N,C,H,W))
        col_w = self.w.reshape(F, HH*WW*C)
        out = col_w@self.store[0] + np.expand_dims(self.b,axis=1)
        out = out.reshape(F,Hout, Wout, N).transpose(3,0,1,2)

        return out

    def backward(self, delta):
        """
        Backward pass of convolutional layer.
        
        delta: gradients from layer above
        dx: gradients that are propagated to layer below
        """
        ######################################################
        ######## REPLACE NEXT PART WITH YOUR SOLUTION ########
        ######################################################
        x,(N, C, H, W) = self.store
        F, C, HH, WW = self.filtersize 

        Wout = int((W - self.filtersize[3]+2*self.pad)/self.stride+1)
        Hout = int((H - self.filtersize[2]+2*self.pad)/self.stride+1)
        
        #Update bias
        self.db = np.sum(delta,axis=(0,2,3)).reshape(F)

        #Reshape delta so that it can be used in vectorized convolution
        delta_flat = delta.transpose(1,2,3,0).reshape(F,N*Wout*Hout)

        #Create column of weights
        col_w = self.w.reshape(F, HH*WW*C)

        #Find delta input
        dx = col_w.T @ delta_flat
        dx = utils.col2im_indices(dx, (N,C,H,W), HH, WW, self.pad, self.stride)

        #Find delta weights
        self.dw = (delta_flat @ x.T).reshape(self.w.shape)

        # Updates the weights and bias using the computed gradients
        self.w -= update_param(self.dw)
        self.b -= update_param(self.db)

        return dx


class MaxPoolingLayer(Layers):
    """
    Implementation of MaxPoolingLayer.
    pool_r, pool_c: integers that denote pooling window size along row and column direction
    stride: integer that denotes with what stride the window is applied
    """
    def __init__(self, pool_r, pool_c, stride):
        self.pool_r = pool_r
        self.pool_c = pool_c
        self.stride = stride

    def forward(self, x):
        """
        Forward pass.
        x: Input tensor of form (NxCxHxW)
        out: Output tensor of form NxCxH_outxW_out
        N: Batch size
        C: Nr of channels
        H, H_out: Input and output heights
        W, W_out: Input and output width
        """
        N, C, H, W = x.shape

        #Calculate output shape
        Hout = (H - self.pool_r) // self.stride + 1
        Wout = (W - self.pool_c) // self.stride + 1

        #Reshape all channels into individuall images
        x = x.reshape(N * C, 1, H, W)
        #Create a list of (pool_c*pool_r, C*H*W) which can be used to find max
        x_col = utils.im2col_indices(x, self.pool_c,self.pool_r, 0, self.stride)
        idx = np.argmax(x_col, axis=0)

        self.store = (idx, x_col.shape, (N,C,H,W))

        #Reshape column back into output image
        out = np.reshape(x_col[idx, range(len(idx))],(Hout, Wout, N, C)).transpose(2,3,0,1)

        return out

    def backward(self, delta):
        """
        Backward pass.
        delta: loss derivative from above (of size NxCxH_outxW_out)
        dX: gradient of loss wrt. input (of size NxCxHxW)
        """

        idx, x_col_shape, (N,C,H,W) = self.store
        zeros = np.zeros(x_col_shape)

        delta_flat = delta.transpose(2,3,0,1).reshape(-1)
        zeros[idx, range(len(idx))] = delta_flat
        dx = utils.col2im_indices(zeros, (N*C, 1,H,W), self.pool_c, self.pool_r, 0, self.stride)

        return dx.reshape(N,C,H,W)


class LSTMLayer(Layers):
    """
    Implementation of a LSTM layer.

    dim_in: Integer indicating input dimension
    dim_hid: Integer indicating hidden dimension
    wx: Weight tensor for input to hidden mapping (dim_in, 4*dim_hid)
    wh: Weight tensor for hidden to hidden mapping (dim_hid, 4*dim_hid)
    b: Bias vector of layer (4*dim_hid)
    """
    def __init__(self, dim_in, dim_hid):
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.wx = np.random.normal(0, 0.1, (dim_in, 4*dim_hid))
        self.wh = np.random.normal(0, 0.1, (dim_hid, 4*dim_hid))
        self.b = np.random.normal(0, 0.1, (4*dim_hid,))

    def forward_step(self, x, h, c):
        """
        Implementation of a single forward step (one timestep)
        x: Input to layer (Nxdim_in) where N=#samples in batch and dim_in=feature dimension
        h: Hidden state from previous time step (Nxdim_hid) where dim_hid=#hidden units
        c: Cell state from previous time step (Nxdim_hid) where dim_hid=#hidden units
        next_h: Updated hidden state(Nxdim_hid)
        next_c: Updated cell state(Nxdim_hid)
        cache: A tuple where you can store anything that might be useful for the backward pass
        """
        
        #Reshape X so that it matches h
        x = x.reshape(h.shape[0],-1)
        #Concatenate earlier weights
        hx = np.concatenate((h,x), 1)
        #Concatenate weight matricies
        whwx = np.concatenate((self.wh, self.wx), 0)
        
        #Multiply input and weight matrix
        out = (hx @ whwx + self.b).T

        #Sigmoid on forget, output and update gates
        sout = sigmoid(out[:3*self.dim_hid,:])
        update_gate = sout[:self.dim_hid,:].T
        forget_gate = sout[self.dim_hid:2*self.dim_hid, :].T
        output_gate = sout[2*self.dim_hid:3*self.dim_hid, :].T
        #Tanh, on update
        tanout = np.tanh(out[3*self.dim_hid:,:]).T
        #Calculate C based on gate outputs
        next_c = c * forget_gate + (update_gate * tanout)
        #Calculate output based on C and outputgate
        next_h = output_gate*np.tanh(next_c)

        cache = (next_h, next_c)

        return next_h, next_c, cache

    def backward_step(self, delta_h, delta_c, store):
        """
        Implementation of a single backward step (one timestep)
        delta_h: Upstream gradients from hidden state
        delta_h: Upstream gradients from cell state
        store:
          hn: Updated hidden state from forward pass (Nxdim_hid) where dim_hid=#hidden units
          x: Input to layer (Nxdim_in) where N=#samples in batch and dim_in=feature dimension
          h: Hidden state from previous time step (Nxdim_hid) where dim_hid=#hidden units
          cn: Updated cell state from forward pass (Nxdim_hid) where dim_hid=#hidden units
          c: Cell state from previous time step (Nxdim_hid) where dim_hid=#hidden units
          cache: Whatever was added to the cache in forward pass
        dx: Gradient of loss wrt. input
        dh: Gradient of loss wrt. previous hidden state
        dc: Gradient of loss wrt. previous cell state
        dwh: Gradient of loss wrt. weight tensor for hidden to hidden mapping
        dwx: Gradient of loss wrt. weight tensor for input to hidden mapping
        db: Gradient of loss wrt. bias vector
        """
        hn, x, h, cn, c, cache = store

        ######################################################
        ######## REPLACE NEXT PART WITH YOUR SOLUTION ########
        ######################################################
        dx = np.random.random_sample(x.shape)
        dh = np.random.random_sample(h.shape)
        dc = np.random.random_sample(c.shape)
        dwh = np.random.random_sample(self.wh.shape)
        dwx = np.random.random_sample(self.wx.shape)
        db = np.random.random_sample(self.b.shape)
        ######################################################
        ######################################################
        ######################################################

        return dx, dh, dc, dwh, dwx, db


class WordEmbeddingLayer(Layers):
    """
    Implementation of WordEmbeddingLayer.
    """
    def __init__(self, vocab_dim, embedding_dim):
        self.w = np.random.normal(0, 0.1, (vocab_dim, embedding_dim))
        self.dw = None

    def forward(self, x):
        """
        Forward pass.
        Look-up embedding for x of form (NxTx1) where each element is an integer indicating the word id.
        N: Number of words in batch. 
        T: Number of timesteps.
        Output: (NxTxE) where E is embedding dimensionality.
        """
        self.store = x
        return self.w[x,:]

    def backward(self, delta):
        """
        Backward pass. Update embedding matrix.
        Delta: Loss derivative from above
        """
        x = self.store
        self.dw = np.zeros(self.w.shape)
        np.add.at(self.dw, x, delta)
        self.w -= update_param(self.dw)
        return 0


"""
Activation functions
"""
class SoftmaxLossLayer(Layers):
    """
    Implementation of SoftmaxLayer forward pass with cross-entropy loss.
    """
    def forward(self, x, y):
        ex = np.exp(x-np.max(x, axis=1, keepdims=True))
        y_hat = ex/np.sum(ex, axis=1, keepdims=True)
        m = y.shape[0]
        log_likehood = -np.log(y_hat[range(m), y.astype(int)])
        loss = np.sum(log_likehood) / m

        d_out = y_hat
        d_out[range(m), y.astype(int)] -= 1
        d_out /= m

        return loss, d_out

class SoftmaxLayer(Layers):
    """
    Implementation of SoftmaxLayer forward pass.
    """
    def forward(self, x):
        ex = np.exp(x-np.max(x, axis=1, keepdims=True))
        y_hat = ex/np.sum(ex, axis=1, keepdims=True)
        return y_hat


class ReluLayer(Layers):
    """
    Implementation of relu activation function.
    """
    def forward(self, x):
        """
        x: Input to layer. Any dimension.
        """
        ######################################################
        ######## REPLACE NEXT PART WITH YOUR SOLUTION ########
        ######################################################
        self.store = x
        
        #Relu is the maximum of 
        out = np.maximum(0,x)
        ######################################################
        ######################################################
        ######################################################
        return out

    def backward(self, delta):
        """
        delta: Loss derivative from above. Any dimension.
        """
        x = self.store
        #The derivative relu is zero for x < 0 and 1 otherwise
        dx = delta * np.where(x<0, 0,1)
        return dx
