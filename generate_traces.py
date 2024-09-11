import tensorflow as tf
import numpy as np

from processtransformer import constants
from processtransformer.data import loader
from processtransformer.models import transformer

# config
NUM_TO_GENERATE = 10

# load data
data_loader = loader.LogsDataLoader(name='helpdesk', dir_path='./datasets')

_, _, x_word_dict, y_word_dict, max_case_length, vocab_size, num_output = data_loader.load_data(task=constants.Task.NEXT_ACTIVITY)
inverse_x_word_dict = { i: a for a, i in x_word_dict.items() }
inverse_y_word_dict = { i: a for a, i in y_word_dict.items() }

# load model
model = transformer.get_next_activity_model(
  max_case_length=max_case_length, 
  vocab_size=vocab_size,
  output_dim=num_output
)
model.load_weights('./models/helpdesk/next_activity_ckpt').expect_partial() # load weights, silence warnings


# generate
for i in range(NUM_TO_GENERATE):
  print(f'\n\nGenerating trace {i+1}...')

  start_of_trace = np.zeros((1, max_case_length))
  start_of_trace[0, -1] = x_word_dict['assign-seriousness']
  trace = start_of_trace

  for j in range(max_case_length-1):
    # predict next activity
    next_activity_pred = model.predict(trace, verbose=0)
    next_activity = tf.random.categorical(next_activity_pred, 1).numpy()[0][0] # random sampling
    # next_activity = np.argmax(next_activity_pred[0]) # argmax
    next_activity_name = inverse_y_word_dict[next_activity]

    # append next activity to trace
    trace = np.roll(trace, -1, axis=1)
    trace[0, -1] = x_word_dict[next_activity_name]

    # stop on EOT
    if next_activity_name == 'closed':
      break

  # print trace
  for activity in trace[0]:
    if inverse_x_word_dict[activity] == '[PAD]': continue

    print(inverse_x_word_dict[activity], end=', ')
