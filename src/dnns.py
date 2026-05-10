import tensorflow as tf


class GraphNetworkLayer(tf.keras.layers.Layer):

    def __init__(self,edge_in_dim, node_in_dim, edge_out_dim, node_out_dim, M, **kwargs):
        super(GraphNetworkLayer, self).__init__(**kwargs)

        self.edge_in_dim = edge_in_dim
        self.node_in_dim = node_in_dim
        self.edge_out_dim = edge_out_dim
        self.node_out_dim = node_out_dim
        self.M = M

    def build(self, input_shape):
        
        self.w_edge = self.add_weight(
            shape=(self.edge_in_dim + 2* self.node_in_dim,self.edge_out_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="w_edge"
        )

        self.b_edge = self.add_weight(
            shape=(self.edge_out_dim,),
            initializer="zeros",
            trainable=True,
            name="b_edge"
        )

        self.w_node = self.add_weight(
            shape=(self.node_in_dim + self.edge_out_dim,self.node_out_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="w_node"
        )

        self.b_node = self.add_weight(
            shape=(self.node_out_dim,),
            initializer="zeros",
            trainable=True,
            name="b_node"
        )

    def call(self, node_features, edge_features, node_idxs_on_edges, num_nodes):

        """
        node_features of shape (N, 5)
             where we have (x_coord, y_coord, out_degree, num_plows_in, num_plows_out) as features 
        
        edge_features of shape (E, 5)
            where we have (snow level, traffic level, length of edge, max speed, num lanes )

        node_idxs_on_edges of shape (E, 2)
            denoted indicies of nodes that are on the ends of the given edge, attribure of road_network

        num_nodes: number of nodes in the network
            
        """

        source_node_features = tf.gather(node_features, node_idxs_on_edges[:,0], axis = 1)
        destination_node_features =tf.gather(node_features, node_idxs_on_edges[:,1], axis = 1)
        edge_features = tf.concat([edge_features,source_node_features,destination_node_features], axis = -1)
        edge_feature_out = tf.nn.relu(tf.matmul(edge_features, self.w_edge) + self.b_edge)
        
        # aggregated_edges = tf.math.unsorted_segment_sum(edge_feature_out, node_idxs_on_edges[:, 1], num_segments=num_nodes)
        aggregated_edges = tf.einsum('ne,bed->bnd', self.M, edge_feature_out)
        node_features_out =  tf.nn.relu(tf.matmul( tf.concat([node_features, aggregated_edges], axis = -1), self.w_node) + self.b_node)

        return edge_feature_out, node_features_out

        
class GNEncoder(tf.keras.Model):

    def __init__(self, num_layers, edge_in_dim, node_in_dim, edge_out_dim, node_out_dim, M,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.gn_layers = [GraphNetworkLayer(edge_in_dim, node_in_dim, edge_out_dim, node_out_dim, M) for _ in range(num_layers)]
        
        self.mlp_to_global = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(edge_out_dim + node_out_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        ])

    def call(self, node_features, edge_features, node_idxs_on_edges, num_nodes):
        for layer in self.gn_layers:
            edge_features, node_features = layer(node_features, edge_features,  node_idxs_on_edges, num_nodes)
        
        pooled_edge_features = tf.reduce_mean(edge_features, axis = 1)
        poooled_node_features = tf.reduce_mean(node_features, axis = 1)
        global_representation = self.mlp_to_global(tf.concat([pooled_edge_features, poooled_node_features], axis = -1))

        return node_features, edge_features, global_representation
    



class Actor(tf.keras.Model):

    def __init__(self, max_num_edges, num_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = tf.keras.layers.Dense(64, activation='gelu')
        self.dense2 = tf.keras.layers.Dense(64, activation='gelu')
        self.dense3 = tf.keras.layers.Dense(64, activation='gelu')
        self.out = tf.keras.layers.Dense(1)

    def call(self, x):
        mask = tf.reduce_all(tf.equal(x, 0.0), axis=-1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        logits = tf.squeeze(self.out(x), axis=-1)
        logits = logits + tf.cast(mask, tf.float32) * -1e9
        return tf.nn.softmax(logits, axis=-1)
    
    
    
class Critic(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = tf.keras.layers.Dense(128, activation='gelu')
        self.dense2 = tf.keras.layers.Dense(128, activation='gelu')
        self.out = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.out(x)
        return x
