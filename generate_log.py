import os
import datetime
import pm4py
import pandas as pd
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from processtransformer import constants
from processtransformer.data import loader
from processtransformer.models import transformer

# constants
PADDING = '[PAD]'
EOT = '[EOT]'

TRACE_KEY = 'case:concept:name'
ACTIVITY_KEY = 'concept:name'
TIMESTAMP_KEY = 'time:timestamp'

# config
MODEL_CHECKPOINT_PATH = os.path.join('models', 'helpdesk', 'next_activity_ckpt')

NUM_TRACES = 4580
OUTPUT_PATH = os.path.join('generated')



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
model.load_weights(MODEL_CHECKPOINT_PATH).expect_partial() # load weights, silence warnings

# generate
generated_data = []

for i in tqdm(range(NUM_TRACES), desc='Generating traces'):
  error = False

  start_of_trace = np.zeros((1, max_case_length))
  start_of_trace[0, -1] = x_word_dict[EOT]
  trace = start_of_trace

  for j in range(max_case_length):
    # predict next activity
    try:
      next_activity_pred = model.predict(trace, verbose=0)
    except Exception as e:
      print(f"An error occurred during model prediction, while generating trace {i}: {e}")
      error = True
      break
    
    next_activity = tf.random.categorical(next_activity_pred, 1).numpy()[0][0] # random sampling
    # next_activity = np.argmax(next_activity_pred[0]) # argmax
    next_activity_name = inverse_y_word_dict[next_activity]

    # append next activity to trace
    trace = np.roll(trace, -1, axis=1)
    trace[0, -1] = x_word_dict[next_activity_name]

    # stop on EOT
    if next_activity_name == EOT:
      break

  # if there was an error in the generation process, skip this trace and retry
  if error:
    i -= 1
    continue

  # add generated trace to generated_data
  for activity in trace[0]:
    if inverse_x_word_dict[activity] == PADDING: continue
    if inverse_x_word_dict[activity] == EOT: continue

    row = {
      TRACE_KEY: f'GEN-{i}',
      ACTIVITY_KEY: inverse_x_word_dict[activity],
      TIMESTAMP_KEY: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    generated_data.append(row)


print('\n\n')

# save generated data to log
generated_log = pd.DataFrame(generated_data, columns=[TRACE_KEY, ACTIVITY_KEY, TIMESTAMP_KEY])
generated_log[TIMESTAMP_KEY] = pd.to_datetime(generated_log[TIMESTAMP_KEY])
generated_log[TRACE_KEY] = generated_log[TRACE_KEY].astype(str)

generated_log_path = os.path.join(OUTPUT_PATH, 'generated.xes')
pm4py.write_xes(generated_log, generated_log_path, case_id_key=TRACE_KEY)
generated_log.to_csv(generated_log_path.replace('.xes', '.csv'), sep=',', index=False)