#coding=utf-8
import time
import math
import numpy as np
import tensorflow as tf

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float   ('learning_rate',     0.00003, 'Initial learning rate.')
tf.app.flags.DEFINE_integer ('steps_to_validate', 500,     'Steps to validate and print loss')
tf.app.flags.DEFINE_integer ("train_steps",       20000,   "Number of (global) training steps to perform")
# For distributed
tf.app.flags.DEFINE_string ("ps_hosts",           "",      "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string ("worker_hosts",       "",      "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string ("job_name",           "",      "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index",         0,       "Index of task within the job")
tf.app.flags.DEFINE_boolean("issync",             True,    "Whether to use Synchronise mode")
tf.app.flags.DEFINE_boolean("use_alienv",         False,   "Whether to use Aliyun environment")
tf.app.flags.DEFINE_string ('device_ids',         '0',     'devices used on every node, use , to seperate device id. use -1 to select all devices on a node')

#Graph parameteres:
tf.app.flags.DEFINE_integer("hidden_units",       100,     "Number of units in the hidden layer of the NN")
tf.app.flags.DEFINE_integer("in_dim",             100,     "Input dimension of the NN")
tf.app.flags.DEFINE_integer("out_dim",            10,      "Output dimension of the NN")
tf.app.flags.DEFINE_integer("batch_size",         20,      "Training batch size")

# Hyperparameters
learning_rate = FLAGS.learning_rate
steps_to_validate = FLAGS.steps_to_validate

def set_ps_worker_from_env():

  if FLAGS.job_name is None or FLAGS.job_name == "":
    if os.getenv("JOB_NAME") is not None:
      FLAGS.job_name = os.getenv("JOB_NAME")
    else:
      raise ValueError("Must specify an explicit `job_name`")
  if FLAGS.task_index is None or FLAGS.task_index =="":
    if os.getenv("JOB_INDEX") is not None:
      FLAGS.task_index = int(os.getenv("JOB_INDEX"))
    else:
      raise ValueError("Must specify an explicit `task_index`")

  print("job name = %s" % FLAGS.job_name)
  print("task index = %d" % FLAGS.task_index)

  #Construct the cluster and start the server
  if FLAGS.ps_hosts is None or FLAGS.ps_hosts =="":
    if os.getenv("PS_HOSTS") is not None:
      FLAGS.ps_hosts = os.getenv("PS_HOSTS")
    else:
      raise ValueError("Failed to find PS hosts info.")
  if FLAGS.worker_hosts is None or FLAGS.worker_hosts =="":
    if os.getenv("WORKER_HOSTS") is not None:
      FLAGS.worker_hosts = os.getenv("WORKER_HOSTS")
    else:
      raise ValueError("Failed to find Worker hosts info.")

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def get_result(data,target,optimizer):
  #define the graph:
  # Ops: located on the worker specified with FLAGS.task_index
  # hidden layer - layer_linear
  with tf.name_scope('layer_linear'):
    with tf.name_scope('weights'):
      # Variables of the hidden layer
      hid_w = tf.get_variable('hid_w',[FLAGS.in_dim, FLAGS.hidden_units])
      #variable_summaries(hid_w)
    with tf.name_scope('bias'):
      hid_b = tf.get_variable('hid_b',[FLAGS.hidden_units])
      #variable_summaries(hid_b)
    with tf.name_scope('Wx_plus_b'):
      hid_lin = tf.nn.xw_plus_b(data, hid_w, hid_b)
      #tf.summary.histogram('pre_activations', hid_lin)
    # activation - Relu
    hid = tf.nn.relu(hid_lin)
    #tf.summary.histogram('activations', hid)

  # softmax layer
  with tf.name_scope('softmax_layer'):
    # Variables of the softmax layer
    sm_w = tf.get_variable('sm_w',[FLAGS.hidden_units, FLAGS.out_dim])
    sm_b = tf.get_variable('sm_b',[FLAGS.out_dim])
    y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b), name="y")
  
  # lose function - cross_entropy
  with tf.name_scope('cross_entropy'): 
    cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
  #tf.summary.scalar('cross_entropy', cross_entropy)
  
  #with tf.name_scope('train'):
  #  gradients = optimizer.compute_gradients(cross_entropy)

  return cross_entropy,hid_w,hid_b

def average_gradients(tower_gradients,cpu_device):
    r'''
    A routine for computing each variable's average of the gradients obtained from the GPUs.
    Note also that this code acts as a syncronization point as it requires all
    GPUs to be finished with their mini-batch before it can run to completion.
    '''
    # List of average gradients to return to the caller
    average_grads = []

    # Run this on cpu_device to conserve GPU memory
    with tf.device(cpu_device):
        # Loop over gradient/variable pairs from all towers
        idx=0
        for grad_and_vars in zip(*tower_gradients):
            # Introduce grads to store the gradients for the current variable
            grads = []
            # Loop over the gradients for the current variable
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Create a gradient/variable tuple for the current variable with its average gradient
            grad_and_var = (grad, grad_and_vars[0][1])

            # Add the current tuple to average_grads
            average_grads.append(grad_and_var)
            idx+=1

    # Return result to caller
    return average_grads

def main(_):

  if FLAGS.use_alienv:
    set_ps_worker_from_env()
  
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)

  issync = FLAGS.issync
  #name of each node
  worker_device = '/job:%s/task:%d' % (FLAGS.job_name, FLAGS.task_index)
  node_ids=FLAGS.device_ids.split(",")
  available_devices = [worker_device + "/gpu:"+idx for idx in node_ids]
  cpu_device = worker_device + '/cpu:0'

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    with tf.device(tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % FLAGS.task_index,
                    cluster=cluster)):
      global_step = tf.Variable(0, name='global_step', trainable=False)

      # Input placeholders
      with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, FLAGS.in_dim], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, FLAGS.out_dim], name='y-input')
      
      with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        if issync:
          #update in sync mode
          optimizer = tf.train.SyncReplicasOptimizer(optimizer,
                replicas_to_aggregate=len(worker_hosts),
                total_num_replicas=len(worker_hosts),
                use_locking=True)

      #define the graph and get results
      tower_cross_entropy=[]
      tower_gradients=[]
      tower_hid_w=[]
      tower_hid_b=[]
      
      sub_datas=tf.split(x,num_or_size_splits=len(available_devices),axis=0)
      sub_targets=tf.split(y_,num_or_size_splits=len(available_devices),axis=0)
      with tf.variable_scope(tf.get_variable_scope()):
        for i in range(len(available_devices)):
          device = tf.train.replica_device_setter(worker_device=available_devices[i], cluster=cluster)
          with tf.device(device):
            # Create a scope for all operations of tower i
            with tf.name_scope('tower_%d' % i) as scope:
              cross_entropy,hid_w,hid_b=get_result(sub_datas[i],sub_targets[i],optimizer)
              tf.get_variable_scope().reuse_variables()
              tower_cross_entropy.append(cross_entropy)
              gradients=optimizer.compute_gradients(cross_entropy)
              tower_gradients.append(gradients)
              tower_hid_w.append(hid_w)
              tower_hid_b.append(hid_b)
      avg_tower_gradients = average_gradients(tower_gradients,cpu_device)

      loss=tf.reduce_mean(tower_cross_entropy, 0)
      #for gradient in avg_tower_gradients
      #    variable_summaries(gradient)
      tf.summary.scalar('cross_entropy', loss)

      with tf.name_scope('train'):
        if issync:
          #update in sync mode
          train_op = optimizer.apply_gradients(avg_tower_gradients,
                                         global_step=global_step)
          init_token_op = optimizer.get_init_tokens_op()
          chief_queue_runner = optimizer.get_chief_queue_runner()
        else:
          #update in async mode
          train_op = optimizer.apply_gradients(avg_tower_gradients,
                                         global_step=global_step)

      init_op = tf.initialize_all_variables()
      saver = tf.train.Saver()
      summary_op = tf.summary.merge_all()
 
      sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                              logdir="./checkpoint/",
                              init_op=init_op,
                              summary_op=None,
                              saver=saver,
                              global_step=global_step,
                              save_model_secs=600)
      
      with sv.prepare_or_wait_for_session(server.target) as sess:
        start_time=time.time()
        #if sync mode:
        if FLAGS.task_index == 0 and issync:
          sv.start_queue_runners(sess, [chief_queue_runner])
          sess.run(init_token_op)
        step = 0
        while  step < FLAGS.train_steps:
          fake_data = np.random.rand(len(available_devices)*FLAGS.batch_size,FLAGS.in_dim)
          fake_target = np.random.rand(len(available_devices)*FLAGS.batch_size,FLAGS.out_dim)
          _, loss_v, step = sess.run([train_op,loss,global_step], 
            feed_dict={x:fake_data, y_:fake_target})

          if step % steps_to_validate == 0:
            w,b = sess.run([tower_hid_w[0],tower_hid_b[0]])
            print("[idx_%d]step: %d/%d, weight[0][0]: %f, biase[0]: %f, loss: %f " 
              %(FLAGS.task_index,step,FLAGS.train_steps,w[0][0],b[0],loss_v))
        end_time=time.time()
        print("distributed train use time: "+str(end_time-start_time))
        
      sv.stop()

      #end_time=time.time()
      #print("distributed train use time: "+str(end_time-start_time))

def loss(label, pred):
  return tf.square(label - pred)

if __name__ == "__main__":
  tf.app.run()