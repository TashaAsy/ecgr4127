#include <Arduino.h>
#include <TensorFlowLite.h>
#include "sin_predictor.h"
#include "Wire.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"


#define INPUT_BUFFER_SIZE 64
#define OUTPUT_BUFFER_SIZE 64
#define INT_ARRAY_SIZE 8

// put function declarations here:
int string_to_array(char *in_str, int *int_array);
void print_int_array(int *int_array, int array_len);
int sum_array(int *int_array, int array_len);

char received_char = (char)NULL;              
int chars_avail = 0;                    // input present on terminal
char out_str_buff[OUTPUT_BUFFER_SIZE];  // strings to print to terminal
char in_str_buff[INPUT_BUFFER_SIZE];    // stores input from terminal
int input_array[INT_ARRAY_SIZE];        // array of integers input by user

int in_buff_idx=0; // tracks current input location in input buffer
int array_length=0;
int array_sum=0;

// Globals, for compatibility with Arduino-style sketches.
namespace {
static tflite::ErrorReporter* error_reporter = nullptr;
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;


// to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over unsigned model format.
// this means that input images must be converted from unsigned to signed.
// the easiest and quickest way to convert from unsigned to signed
// 8-bit integers is to subtract 128 from the unsigned value to get a signed value.

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 136 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
}

// Measure and print time for printing
void measure_and_print() {
  unsigned long t0, t1, t2;
  t0 = micros();
  Serial.println("Test statement");
  t1 = micros();
  interpreter->Invoke();
  t2 = micros();
  long t_print = t1 - t0;
  long t_infer = t2 - t1;
  Serial.print("Print time: ");
  Serial.print(t_print);
  Serial.print(". Inference time:");
  Serial.println(t_infer);
}

void setup() {
  // put your setup code here, to run once:
  delay(5000);
  // Arduino does not have a stdout, so printf does not work easily
  // So to print fixed messages (without variables), use 
  // Serial.println() (appends new-line)  or Serial.print() (no added new-line)
  Serial.println("Test Project waking up");
  memset(in_str_buff, (char)0, INPUT_BUFFER_SIZE*sizeof(char)); 

  // set up logging.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // map model inot a usable data structure. doesn't involve any
  // copying or parsing
  model = tflite::GetModel(sin_predictor_tflite);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                        "Model provided is schema version %d not equal "
                        "to supported version %d.",
                        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  // pull in only the operation implementations we need.

  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddRelu();
  micro_op_resolver.AddAdd();
  micro_op_resolver.AddMul();
  micro_op_resolver.AddSub();
  micro_op_resolver.AddMean();

    // build an interpreter to run the model with.
    // NOLINTNEXTLINE(runtime-global-variables)
    static tflite::MicroInterpreter static_interpreter(
          model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    // allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
      return;
    }

    // get info about the memory area to use for the model's input
    input = interpreter->input(0);
}

void loop() {
  // put your main code here, to run repeatedly:

  // check if characters are avialble on the terminal input
  chars_avail = Serial.available(); 
  if (chars_avail > 0) {
    received_char = Serial.read(); // get the typed character and 
    Serial.print(received_char);   // echo to the terminal

    in_str_buff[in_buff_idx++] = received_char; // add it to the buffer
    if (received_char == 13) { // 13 decimal = newline character
      // user hit 'enter', so we'll process the line.
      Serial.print("About to process line: ");
      Serial.println(in_str_buff);
      // end code

      // Now clear the input buffer and reset the index to 0
      memset(in_str_buff, (char)0, INPUT_BUFFER_SIZE*sizeof(char)); 
      in_buff_idx = 0;
    }
    else if (in_buff_idx >= INPUT_BUFFER_SIZE) {
      memset(in_str_buff, (char)0, INPUT_BUFFER_SIZE*sizeof(char)); 
      in_buff_idx = 0;
    }    
  }

  // process and print out the array
  array_length = string_to_array(in_str_buff, input_array);
  if (array_length != 7) {
    Serial.println("Error: Please enter exactly 7 numbers.");
  } else {
    // Convert form int to int, model needs 7 inputs
    for (int i = 0; i < array_length; i++) {
      // convert int input to the expected format
      input->data.int8[i] = static_cast<int8>(input_array[i]);
    }
    // run the model on the input and make sure it succeeds
    if (kTfLiteOk != interpreter->Invoke()) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
    } else {
      // the output is a single float
      TfLiteTensor* output = interpreter->output(0);
      int8 prediction = output->data.int8[0];
      Serial.print("Model prediction: ");
      Serial.println(prediction);
    }
  }
}

int string_to_array(char *in_str, int *int_array) {
  int num_integers=0;
  char *token = strtok(in_str, ",");
  
  while (token != NULL) {
    int_array[num_integers++] = atoi(token);
    token = strtok(NULL, ",");
    if (num_integers >= INT_ARRAY_SIZE) {
      break;
    }
  }
  
  return num_integers;
}

void print_int_array(int *int_array, int array_len) {
  int curr_pos = 0; // track where in the output buffer we're writing

  sprintf(out_str_buff, "Integers: [");
  curr_pos = strlen(out_str_buff); // so the next write adds to the end
  for(int i=0;i<array_len;i++) {
    // sprintf returns number of char's written. use it to update current position
    curr_pos += sprintf(out_str_buff+curr_pos, "%d, ", int_array[i]);
  }
  sprintf(out_str_buff+curr_pos, "]\r\n");
  Serial.print(out_str_buff);
}

int sum_array(int *int_array, int array_len) {
  int curr_sum = 0; // running sum of the array

  for(int i=0;i<array_len;i++) {
    curr_sum += int_array[i];
  }
  return curr_sum;
}