# !pip install transformers==2.8.0
# !pip install torch==1.4.0
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    device = torch.device('cpu')
    text = request.form.values()
    preprocessed_text = text.strip().replace('\n','')
    t5_input_text =  preprocessed_text
    tokenized_text = tokenizer.encode(t5_input_text, return_tensors='pt', max_length=512).to(device)
    summary_ids = model.generate(tokenized_text, min_length=30, max_length=120)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return render_template('index.html', prediction_text= summary )

if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
