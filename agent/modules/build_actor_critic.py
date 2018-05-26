import tensorflow as tf
import math

class Build_network(object):

    def __init__(self,sess,config,name):
        self.name=name
        self.sess=sess
        self.trainable=False if name.split('_')=='target' else True
        with tf.name_scope(name):
            self.state=tf.placeholder(tf.float32,[None,config.state_dim])
            layers=[[config.state_dim,config.layers[0]]]
            for layer in zip(config.layers[:-1],config.layers[1:]):
                layers.append(list(layer))
            if name[0]=='a':
                layers.append([config.layers[-1],config.action_dim])
            else:
                layers.append([config.layers[-1],1])
                layers[1][0]+=config.action_dim
                self.action=tf.placeholder(tf.float32,[None,config.action_dim])
            for idx,(in_dim,out_dim) in enumerate(layers):
                self.create_variable(in_dim,out_dim,'fc'+str(idx))
            self.var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,name)
            self.variables={var.name:var for var in self.var_list}
            out_=tf.nn.relu(self.layer_fc(self.state,'fc0'))
            out_=tf.concat([out_,self.action],1) if name[0]=='c' else out_
            out_=self.layer_fc(out_,'fc1')
            for layer in range(2,len(layers)):
                out_=self.layer_fc(tf.nn.relu(out_),'fc'+str(layer))
            if name[0]=='a':
                out_0,out_1=tf.split(out_,[1,1],axis=1)
                self.out_=tf.concat([tf.sigmoid(out_0),tf.tanh(out_1)],1)
            else:
                self.out_=out_    

    def evaluate(self,state,action=None):
        return self.sess.run(self.out_, \
                             feed_dict={self.state:state} if action==None else \
                                       {self.state:state,self.action:action})    

    def layer_fc(self,in_,layer):
        return tf.matmul(in_,self.variables[self.name+'/'+layer+'/w:0'])+ \
                             self.variables[self.name+'/'+layer+'/b:0']    

    def create_variable(self,in_dim,out_dim,name):
        with tf.name_scope(name):
            tf.Variable( \
                tf.random_uniform( \
                    [in_dim,out_dim],-1/math.sqrt(in_dim),1/math.sqrt(in_dim)), \
                name='w',trainable=self.trainable)
            tf.Variable( \
                tf.random_uniform( \
                    [out_dim],-1/math.sqrt(in_dim),1/math.sqrt(in_dim)), \
                name='b',trainable=self.trainable)