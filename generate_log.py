import os
import copy
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
DATASET_NAME = 'bpic2012_a'

NEXT_ACTIVITY_MODEL_PATH = os.path.join('models', DATASET_NAME, 'next_activity_ckpt')
NEXT_TIME_MODEL_PATH = os.path.join('models', DATASET_NAME, 'next_time_ckpt')

NUM_GENERATIONS = 3
NUM_TRACES = 937
OUTPUT_PATH = os.path.join('generated')

# Usually we want to start the generation from the first timestamp of the test log
# or from the last timestamp of the train log
START_TIMESTAMP = datetime.datetime.strptime('27.01.2012 16:58:46', '%d.%m.%Y %H:%M:%S')
# sepsis = 26.10.2014 20:21:00
# bpic2012_a = 27.01.2012 16:58:46
# START_TIMESTAMP = datetime.datetime.now()

# load data
next_activity_data_loader = loader.LogsDataLoader(name=DATASET_NAME, dir_path='./datasets/next_activity')
next_time_data_loader = loader.LogsDataLoader(name=DATASET_NAME, dir_path='./datasets/next_time')

_, _, x_word_dict, y_word_dict, max_case_length, vocab_size, num_output = next_activity_data_loader.load_data(task=constants.Task.NEXT_ACTIVITY)
new_activity_name_to_original_activity_name = next_activity_data_loader.get_new_activity_name_to_original_activity_name()
inverse_x_word_dict = { i: a for a, i in x_word_dict.items() }
inverse_y_word_dict = { i: a for a, i in y_word_dict.items() }

train_df, _, _, _, _, _, _ = next_time_data_loader.load_data(task=constants.Task.NEXT_TIME)
_, _, _, time_scaler, y_scaler = next_time_data_loader.prepare_data_next_time(train_df, x_word_dict, max_case_length)

# load models
next_activity_model = transformer.get_next_activity_model(
  max_case_length=max_case_length, 
  vocab_size=vocab_size,
  output_dim=num_output,
)
next_activity_model.load_weights(NEXT_ACTIVITY_MODEL_PATH).expect_partial() # load weights, silence warnings

next_time_model = transformer.get_next_time_model(
  max_case_length=max_case_length,
  vocab_size=vocab_size,
)
next_time_model.load_weights(NEXT_TIME_MODEL_PATH).expect_partial() # load weights, silence warnings

for num_gen in range(NUM_GENERATIONS):
  print(f'Generation #{num_gen+1}')

  # generate
  generated_data = []

  for i in tqdm(range(NUM_TRACES), desc='Generating traces'):
    error = False

    start_of_trace = np.zeros((1, max_case_length))
    start_of_trace[0, -1] = x_word_dict[EOT]
    trace = start_of_trace
    time = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    times = []
    recent_time_memory = [0.0, 0.0]

    for j in range(max_case_length):
      # predict next activity
      try:
        next_activity_pred = next_activity_model.predict(trace, verbose=0)
      except Exception as e:
        print(f"An error occurred during next_activity_model prediction, while generating trace {i}: {e}")
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

      # predict next time
      try:
        time_input = time_scaler.transform(time).astype(np.float32)

        next_time_pred = next_time_model.predict([trace, time_input], verbose=0)
        next_time_pred = y_scaler.inverse_transform(next_time_pred)
        next_time_pred = int(next_time_pred)
        next_time_pred = 0 if next_time_pred < 0 else next_time_pred

        # keep two latest time_pred in memory
        recent_time_memory[0] = recent_time_memory[1]
        recent_time_memory[1] = next_time_pred

        # update time based on next_time_pred
        # time = [time_passed, recent_time, latest_time]
        # time_passed = sum of all previous next_time_pred
        # recent_time = sum of previous 2 next_time_pred
        # latest_time = previous next_time_pred

        time[0][0] += next_time_pred
        time[0][1] = sum(recent_time_memory)
        time[0][2] = next_time_pred

        times.append(copy.deepcopy(time))
        
      except Exception as e:
        print(f"An error occurred during next_time_model prediction, while generating trace {i}: {e}")
        error = True
        break

    # if there was an error in the generation process, skip this trace and retry
    if error:
      i -= 1
      continue

    activity_number = 0
    # add generated trace to generated_data
    for activity in trace[0]:
      if inverse_x_word_dict[activity] == PADDING: continue
      if inverse_x_word_dict[activity] == EOT: continue

      activity_number += 1

      timestamp = START_TIMESTAMP + datetime.timedelta(days=int(times[activity_number-1][0][0]))

      row = {
        TRACE_KEY: f'GEN-{i}',
        'case:concept:name': f'GEN-{i}',
        ACTIVITY_KEY: new_activity_name_to_original_activity_name[inverse_x_word_dict[activity]],
        'concept:name': new_activity_name_to_original_activity_name[inverse_x_word_dict[activity]],
        TIMESTAMP_KEY: timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        'time:timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
      }

      generated_data.append(row)

  # save generated data to log
  generated_log = pd.DataFrame(generated_data, columns=[TRACE_KEY, ACTIVITY_KEY, TIMESTAMP_KEY])
  generated_log[TIMESTAMP_KEY] = pd.to_datetime(generated_log[TIMESTAMP_KEY])
  generated_log[TRACE_KEY] = generated_log[TRACE_KEY].astype(str)

  generated_log_path = os.path.join(OUTPUT_PATH, f'gen{num_gen+1}.xes')
  pm4py.write_xes(generated_log, generated_log_path, case_id_key=TRACE_KEY)
  generated_log.to_csv(generated_log_path.replace('.xes', '.csv'), sep=';', index=False)

  print('\n')
