import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#mnist dataset is already avialable in input_data library
mnist = input_data.read_data_sets("/tmp/data",one_hot="True")
print("imported dataset")

#no. of neurons in each hidden layer
n_nodes_h1=500
n_nodes_h2=500
n_nodes_h3=500

#total 10 output classes= 0 to 9
classes =10
batch_size=50
x=tf.placeholder("float",[None,784])
y= tf.placeholder("float")

def neural_network_model(data):
    # 784=28*28 is the input size of input layer with the value of each pixel color
    hidden_layer_1= {"weights":tf.Variable(tf.random_normal([784,n_nodes_h1])),
                     "bias":tf.Variable(tf.random_normal([n_nodes_h1]))}
    hidden_layer_2 = {"weights":tf.Variable(tf.random_normal([n_nodes_h1,n_nodes_h2])),
                     "bias":tf.Variable(tf.random_normal([n_nodes_h2]))}
    hidden_layer_3= {"weights":tf.Variable(tf.random_normal([n_nodes_h2,n_nodes_h3])),
                     "bias":tf.Variable(tf.random_normal([n_nodes_h3]))}
    output_layer = {"weights":tf.Variable(tf.random_normal([n_nodes_h3,classes])),
                     "bias":tf.Variable(tf.random_normal([classes]))}
    print("model created")
    # now creating the final neural network after individually creating every layer in above steps
    l1= tf.add(tf.matmul(data,hidden_layer_1["weights"]),hidden_layer_1["bias"])
    l1= tf.nn.relu(l1)
    l2= tf.add(tf.matmul(l1,hidden_layer_2["weights"]),hidden_layer_2["bias"])
    l2= tf.nn.relu(l2)
    l3=  tf.add(tf.matmul(l2,hidden_layer_3["weights"]),hidden_layer_3["bias"])
    l3 = tf.nn.relu(l3)
    output= tf.add(tf.matmul(l3,output_layer["weights"]),output_layer["bias"])
    return output

def train_neural_network(x):
    prediction= neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epoch = 20 
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # for each epoch the model is run for different batches and after each epoch 
        # the cross-validation error gets reduced as it is trained on more examples
        # large epocs can lead to over fitting leading to reduction in test case accuracy while incresing training accuracy.
        for epoch in range(hm_epoch):
            epoch_loss=0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y=mnist.train.next_batch(batch_size)
                _,c =sess.run([optimizer,cost],feed_dict={x:epoch_x, y:epoch_y})
                epoch_loss+=c
            print("Epoch",epoch,"completed of",hm_epoch,"loss",epoch_loss)

        correct =tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy =tf.reduce_mean(tf.cast(correct,'float'))
        print("accuracy=",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))

train_neural_network(x)





