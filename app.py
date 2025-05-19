from flask import Flask, request, jsonify
import numpy as np
import tensorflow.lite as tflite

app = Flask(__name__)

# โหลดโมเดล TFLite
interpreter = tflite.Interpreter(model_path="fall_detection_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('inputs')
        if not data or len(data) != 3:
            return jsonify({"error": "Invalid input format. Expecting 3 values."}), 400

        # เตรียม input shape (ซ้ำ 50 timestep)
        time_steps = 50
        input_sequence = np.tile(data, (time_steps, 1)).astype(np.float32)
        input_array = input_sequence.reshape(1, time_steps, 3)

        interpreter.set_tensor(input_details[0]['index'], input_array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        result = "Fall" if output[0][0] > 0.75 else "No Fall"
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)