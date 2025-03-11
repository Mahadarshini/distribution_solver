# Distribution Solver Application

A web application that can parse and solve questions related to statistical distributions using natural language processing.

## Features

- Parses natural language questions about probability distributions
- Supports multiple distribution types:
  - Normal (Gaussian) distribution
  - Binomial distribution
  - Poisson distribution
  - Uniform distribution
- Calculates various statistical measures:
  - Probability (P(X < k), P(X > k), P(X = k), etc.)
  - Mean
  - Variance
  - Standard deviation
  - Quantiles (for normal distribution)
- Provides step-by-step explanations
- Displays parsed interpretations of questions
- Simple, user-friendly interface

## Installation

### Prerequisites

- Python 3.7+
- Flask
- NumPy
- SciPy
- NLTK

### Setup

1. Clone the repository:
```
git clone https://github.com/yourusername/distribution-solver.git
cd distribution-solver
```

2. Create a virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```
pip install flask flask-cors nltk numpy scipy
```

4. Download required NLTK data:
```
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

5. Run the application:
```
python app.py
```

6. Open your browser and navigate to `http://localhost:5000` to use the application.

## Usage

1. Type or select a statistical distribution question in the input box
2. Click the "Solve Problem" button
3. The application will:
   - Parse and analyze your question
   - Display how it understood your question
   - Calculate and show the solution with an explanation

## Example Questions

- "What is the probability that X is less than 15 in a normal distribution with mean 10 and standard deviation 3?"
- "In a binomial distribution with n=10 and p=0.4, what is the probability of getting at least 6 successes?"
- "Find the mean of a Poisson distribution with lambda = 5."
- "Calculate the probability of X being between 2 and 4 in a uniform distribution from 1 to 6."
- "What is the standard deviation of a normal distribution with mean 50 and standard deviation 5?"

## How It Works

1. **Natural Language Processing**: The application uses NLTK to parse questions and extract key information.
2. **Distribution Detection**: It identifies which distribution (normal, binomial, etc.) is being referenced.
3. **Parameter Extraction**: It extracts numerical parameters for the distribution (mean, standard deviation, etc.).
4. **Query Type Identification**: It determines what to calculate (probability, mean, variance, etc.).
5. **Mathematical Computation**: It uses SciPy and NumPy to perform the calculations.
6. **Result Presentation**: It presents the results with explanations in a user-friendly format.

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS (Tailwind CSS), JavaScript
- **Mathematics**: NumPy, SciPy
- **NLP**: NLTK
- **Math Rendering**: MathJax

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.