from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)


# โหลดโมเดลที่อัปโหลดมา
model = tf.keras.models.load_model('fall_detection_model_Acc.keras')
# model = tf.keras.models.load_model('fall_detection_model_Acc_cnn.keras')

@app.route('/predict', methods=['POST'])
def predict():
    
    try:
        data = request.json.get('inputs')
        
        if not data or len(data) != 3:
            return jsonify({"error": "Invalid input format. Expecting 3 values."}), 400
        
        # เติมลำดับเวลา
        time_steps = 50
        input_sequence = np.tile(data, (time_steps, 1))  # ทำซ้ำ 50 ครั้ง
        input_array = input_sequence.reshape(1, time_steps, 3)

        # ทำการพยากรณ์
        prediction = model.predict(input_array)
        result = "Fall" if prediction[0][0] > 0.75 else "No Fall"

        #return jsonify({"result": result})
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)