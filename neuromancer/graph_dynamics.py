import torch
import torch.nn as nn
from neuromancer.dynamics import SSM 
from neuromancer import blocks
from neuromancer import integrators
from neuromancer.gnn import Replace, GraphFeatureUpdate, GraphFeatures, ArgSequential, graph_interp_u


class GraphTimestepper(SSM):
    DEFAULT_INPUT_KEYS = ["x0", "Yf"]
    DEFAULT_OUTPUT_KEYS = ["reg_error", "X_pred", "Y_pred"]

    def __init__(self,
                 num_node_features,
                 num_edge_features,
                 out_size,
                 latent_size=128,
                 hidden_sizes=[128, 128],
                 message_passing_steps=10,
                 preprocessor=None,
                 graph_encoder=None,
                 integrator=None,
                 decoder=None,
                 fx=None,
                 fx_keys=[],
                 separate_batch_dim = False,
                 add_batch_norm = True,
                 input_key_map={},
                 add_out_keys=[],
                 name="graph_timestepper",
                 device='cpu',):
        """Graph Neural Timestepper composed of a preprocessor, a encoder, processor, and decoder block.

        :param num_node_features: [int] Number of node features after preprocesser
        :param num_edge_features: [int] Number of edge features after preprocesser
        :param out_size: [int] Dimension of output
        :param latent_size: [int], Dimension of latent size per node/edge, defaults to 128
        :param hidden_sizes: [list int], Dimensions of hidden sizes of nodes/edges, defaults to [128]
        :param message_passing_steps: [int], Number of message passing layers, defaults to 10
        :param preprocessor: [nn.Module] Accepts a dictionary and returns a dictionary of tensors with keys [edge_index, node_attr, edge_attr, (opt) batch]. Other key/values will be added to output dict.
        :param graph_encoder: [nn.Module] Input Tensor: (nodes, num_node_features). Output Tensor: (nodes, latent_dim). Defaults to a linear layer.
        :param integrator: [Integrator] Input (x, block, *args, **kwargs). Ouput x_new
        :param decoder: [nn.Module] Input Tensor: (nodes, latent_dim). Output Tensor: (nodes, out_size)
        :param fx: [nn.Module] Module to handle extraneous input given in fx_keys
        :param fx_keys: Key names for extraneous variables.
        :param separate_batch_dim: [bool] True when data should be handled with separate batch and node dimensions
        :param input_key_map: [dict] Remaps the name of the inputs
        :param add_output_keys: [list string] Names of output variables in addition to defaults.
        :param name: [string] Name of module. Prepended to all outputs.
        :param device: [torch.device], Device to run the model on, defaults to 'cpu'
        """
        self.DEFAULT_INPUT_KEYS = self.DEFAULT_INPUT_KEYS
        super().__init__(input_key_map, name)
        self.out_size = out_size
        self.latent_size = latent_size
        self.hidden_sizes = hidden_sizes
        self.message_passing_steps = message_passing_steps
        self.device = device
        self.preprocessor = preprocessor if preprocessor is not None else nn.Identity()
        self.node_features = num_node_features
        self.edge_features = num_edge_features
        self.in_features = self.node_features
        self.out_features = self.out_size
        self.separate_batch_dim = separate_batch_dim
        self.fx = fx
        self.fx_keys = fx_keys
        self.input_keys += self.fx_keys
        
        #Build default graph_encoder consisting of MLPs on node and edge features
        if graph_encoder is None:
            node_encoder = nn.Sequential(
                blocks.MLP(
                    insize=self.node_features,
                    outsize=self.latent_size,
                    nonlin=nn.ReLU,
                    hsizes=self.hidden_sizes),
                nn.LayerNorm(self.latent_size))
            edge_encoder = nn.Sequential(
                blocks.MLP(
                    insize=self.edge_features,
                    outsize=self.latent_size,
                    nonlin=nn.ReLU,
                    hsizes=self.hidden_sizes),
                nn.LayerNorm(self.latent_size))
            graph_encoder = GraphFeatureUpdate(node_encoder, edge_encoder)
        self.encoder = graph_encoder

        #Build Processor as a series of message passing steps
        processor = nn.ModuleList()
        integrator = integrators.integrators.get(integrator, Replace)
        self.batch_norm_processor=add_batch_norm
        if self.batch_norm_processor:
            self.batch_norms = nn.ModuleList()
            self.edge_norms = nn.ModuleList()
        for _ in range(self.message_passing_steps):
            if self.batch_norm_processor:
                self.batch_norms.append(nn.BatchNorm1d(self.latent_size))
                self.edge_norms.append(nn.BatchNorm1d(self.latent_size))
            node_process = nn.Sequential(
                blocks.MLP(
                    insize = self.latent_size * 2,
                    outsize = self.latent_size,
                    nonlin = nn.ReLU,
                    hsizes = self.hidden_sizes,
                ),
                nn.LayerNorm(self.latent_size)
            )
            edge_process = nn.Sequential(
                blocks.MLP(
                    insize = self.latent_size * 3,
                    outsize = self.latent_size,
                    nonlin = nn.ReLU,
                    hsizes = self.hidden_sizes
                ),
                nn.LayerNorm(self.latent_size)
            )
            block = integrator(
                        GraphFeatureUpdate(node_process, edge_process),
                        interp_u=graph_interp_u,
                        h=0.1)
            block.state = lambda x, tq, t, u: GraphFeatures.cat([x, block.interp_u(tq, t, u)], dim=-1)               
            processor.append(block)

        self.processor = processor#ArgSequential(*processor)

        #Defines decoder or creates default MLP
        self.decoder = decoder if decoder is not None else\
                        blocks.MLP(
                            insize=self.latent_size,
                            outsize=self.out_size,
                            nonlin=nn.ReLU,
                            hsizes=self.hidden_sizes)
        self.to(self.device)

    def forward(self, data):
        """Forward pass of the module

        :param data: [{"x0": Tensor, "Yf": Tensor}]
        :return: ["reg_error": Tensor, "latent_state": Tensor, "Y_pred": Tensor]
        """
        nsteps = data[self.input_key_map['Yf']].shape[1]

        X = self.preprocessor(data)
        batch = X.pop('batch', None)
        edge_index = X.pop('edge_index')
        node_attr = X.pop('node_attr')
        edge_attr = X.pop('edge_attr')
        add_outputs = X

        if self.separate_batch_dim: #(batch, nodes, timesteps, ...)
            nsteps = data[self.input_key_map['Yf']].shape[1]
            batch_size, num_nodes = node_attr.shape[:2]
            num_edges = edge_attr.shape[1]
            node_attr, edge_attr, edge_index, batch = self._collate(node_attr, edge_attr, edge_index)

        #Encode
        G = GraphFeatures(node_attr, edge_attr)
        G = self.encode(G)

        #Process
        X = [G]
        for h in range(nsteps):
            G = self.process(G, edge_index, *[data[key][:,h] for key in self.fx_keys])
            X.append(G)

        #Decode
        Y=[]
        for x in X[1:]:
            Y.append(self.decode(x))

        Y=torch.stack(Y, dim=1)
        X=torch.stack([x['node_attr'] for x in X], dim=0)
        X=X.reshape(X.shape[0],-1)

        if self.separate_batch_dim:
            Y = Y.reshape(batch_size, num_nodes, nsteps, self.out_size).permute(0,2,1,3).squeeze(-1)
            X = X.reshape(batch_size, num_nodes, nsteps+1, self.latent_size).permute(0,2,1,3).reshape(batch_size, nsteps+1, -1)

        out = {f'Y_pred_{self.name}': Y, 
               f'X_pred_{self.name}': X,
               f'reg_error_{self.name}': self.reg_error(),
               **add_outputs}
        return out

    def encode(self, G):
        """Encodes graph features into the latent space

        :param G: [Graph_Features]
        :return: [Graph_Features]
        """
        return self.encoder(G)

    def process(self, G, edge_index, *args):
        """Processes a series of message passing steps on the graph.

        :param G: [Graph_Features]
        :param edge_index: [Tensor] Shape (2, edges)
        :param kwargs: Additional input to fx
        :return: [Graph_Features]
        """
        if self.fx:
            temp = self.fx(*args)
            G.node_attr = G.node_attr + self.fx(*args).view(G.node_attr.shape)
            
        for mps in range(len(self.processor)):
            u = {'node_attr':G['node_attr'], 'edge_attr':G['edge_attr'], 'edge_index':edge_index}
            if self.batch_norm_processor:
                G.node_attr = self.batch_norms[mps](G.node_attr)
                G.edge_attr = self.edge_norms[mps](G.edge_attr)
            G = self.processor[mps](G, u)
        return G

    def decode(self, G):
        """Transforms the graph features from the latent space to the output space.

        :param G: [Graph_Features]
        :return: [Graph_Features]
        """
        return self.decoder(G['node_attr'])
            
    def _collate(self, node_attr, edge_attr, edge_index):
        """Takes a set of graph attributes and batches them. 
           Subsequent graphs after the first have their node indices incremented.

        :param node_attr:[list tensor]
        :param edge_attr: [list tensor]
        :param edge_index: [list tensor]
        :return: tensors node_attr (batch*nodes, *), edge_attr (batch_edges, *), edge_index (2, batch*edges), batch (batch*nodes)
        """
        #Assumes (batch, node/edge, features)
        batch_size, num_nodes, node_features = node_attr.shape
        _, num_edges, edge_features = edge_attr.shape

        node_attr = torch.reshape(node_attr, (batch_size * num_nodes, node_features))
        edge_attr = torch.reshape(edge_attr, (batch_size * num_edges, edge_features))
        
        #Handle node and batch index incrementing
        new_edge_index = torch.zeros((2, num_edges*batch_size),dtype=torch.long)
        batches = torch.zeros(batch_size*num_nodes, dtype=torch.long)
        for i in range(batch_size):
            batches[num_nodes*i:num_nodes*(i+1)] = i
            new_edge_index[:, num_edges*i:num_edges*(i+1)] = edge_index + (num_nodes * i)

        return node_attr, edge_attr, new_edge_index, batches

    def _uncollate(self, node_attr, edge_attr, edge_index, batches):
        """Takes a batched set of graph attributes and separates the graphs apart
           Graphs must be the same except for the values of the node and edge attributes

        :param node_attr: [tensor] (batch*nodes, ...)
        :param edge_attr: [tensor] (batch*edges, ...)
        :param edge_index: [tensor] (2, batch*edges)
        :param batches: [tensor] (batch*nodes)
        :return: tensors of the form (batch, ...)
        """
        #Assumes (batch*(nodes/edges), features)
        #Returns (batch, nodes/edges, features)
        batch_size = (batches[-1]+1).item()
        num_nodes = node_attr.shape[0] // batch_size
        num_edges = edge_attr.shape[0] // batch_size

        node_attr = torch.reshape(node_attr, (batch_size, num_nodes, -1))
        edge_attr = torch.reshape(edge_attr, (batch_size, num_edges, -1))
        edge_index = edge_index[:, :num_edges]
        return node_attr, edge_attr, edge_index

    def reg_error(self):
        return sum([k.reg_error() for k in self.children() if hasattr(k, 'reg_error')])
