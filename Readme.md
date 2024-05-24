# Learning Residuals

## Steps to Run
1. **Ensure you have gram2vec in directory.  Clone the Repository**:
   ```sh
   git clone https://github.com/eric-sclafani/gram2vec.git
   pip install gram2vec
   ```

2. **Install Dependencies**:
   Make sure you have Python 3.7 or higher. Install the required packages using pip:
   ```sh
   pip install -r requirements.txt
   ```


3. **Train the Model**:
   Run the training script with the appropriate arguments. For example:
   ```sh
   python src/train_residual.py --model_type roberta  --run_id 1 -p 1 --use_lora True
   ```
