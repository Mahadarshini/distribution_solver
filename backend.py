from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import nltk
import re
import numpy as np
from scipy import stats

app = Flask(__name__)
CORS(app)

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class DistributionSolver:
    def __init__(self):
        self.distribution_types = {
            'normal': self._solve_normal,
            'binomial': self._solve_binomial,
            'poisson': self._solve_poisson,
            'uniform': self._solve_uniform
        }
        
        # Keywords to identify distribution types
        self.distribution_keywords = {
            'normal': ['normal', 'gaussian', 'bell curve', 'mean', 'standard deviation', 'z-score'],
            'binomial': ['binomial', 'success', 'failure', 'trials', 'probability of success'],
            'poisson': ['poisson', 'rate', 'lambda', 'events', 'interval'],
            'uniform': ['uniform', 'equally likely', 'range', 'continuous uniform']
        }
        
        # Keywords to identify what to calculate
        self.calculation_keywords = {
            'probability': ['probability', 'chance', 'likelihood', 'what is the probability', 'find the probability'],
            'mean': ['mean', 'average', 'expected value'],
            'variance': ['variance', 'var'],
            'standard_deviation': ['standard deviation', 'std', 'sd'],
            'quantile': ['quantile', 'percentile', 'quartile'],
            'interval': ['interval', 'between', 'range', 'from', 'to']
        }
    
    def parse_question(self, question):
        """Parse the question to identify distribution type, parameters, and what to calculate"""
        # Tokenize the question
        tokens = nltk.word_tokenize(question.lower())
        
        # Identify distribution type
        distribution_type = None
        for dist, keywords in self.distribution_keywords.items():
            if any(keyword in question.lower() for keyword in keywords):
                distribution_type = dist
                break
        
        # Extract numerical values
        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', question)
        parameters = [float(num) for num in numbers]
        
        # Identify what to calculate
        calculation_type = None
        for calc, keywords in self.calculation_keywords.items():
            if any(keyword in question.lower() for keyword in keywords):
                calculation_type = calc
                break
        
        # Extract inequality symbols for probability calculations
        inequality = None
        if '<' in question:
            inequality = '<'
        elif '>' in question:
            inequality = '>'
        elif '<=' in question or '≤' in question:
            inequality = '<='
        elif '>=' in question or '≥' in question:
            inequality = '>='
        elif 'between' in question.lower() or ('from' in question.lower() and 'to' in question.lower()):
            inequality = 'between'
        
        # Extract the value to compare with in probability calculations
        compare_value = None
        if inequality and inequality != 'between':
            # Find the number that appears after the inequality
            matches = re.finditer(r'[-+]?\d*\.\d+|\d+', question)
            for match in matches:
                # Check if the inequality appears before this number in the text
                if question.find(inequality) < match.start():
                    compare_value = float(match.group())
                    break
        
        # Extract interval bounds for "between" calculations
        interval = None
        if inequality == 'between':
            if len(parameters) >= 2:
                # Assuming the first two numbers are the interval bounds
                interval = [parameters[0], parameters[1]]
        
        return {
            'distribution_type': distribution_type,
            'parameters': parameters,
            'calculation_type': calculation_type,
            'inequality': inequality,
            'compare_value': compare_value,
            'interval': interval
        }
    
    def solve(self, question):
        """Solve the distribution problem based on the parsed question"""
        try:
            parsed = self.parse_question(question)
            
            if not parsed['distribution_type']:
                return {"error": "Could not identify distribution type. Please specify normal, binomial, Poisson, or uniform distribution."}
            
            if not parsed['calculation_type']:
                return {"error": "Could not identify what to calculate. Please specify probability, mean, variance, standard deviation, or quantile."}
            
            solver = self.distribution_types.get(parsed['distribution_type'])
            if not solver:
                return {"error": f"Solving for {parsed['distribution_type']} distribution is not supported yet."}
            
            result = solver(parsed)
            return result
        except Exception as e:
            return {"error": f"Error solving problem: {str(e)}"}
    
    def _solve_normal(self, parsed):
        """Solve problems related to normal distribution"""
        params = parsed['parameters']
        
        # Extract mean and standard deviation
        if len(params) >= 2:
            mean = params[0]
            std_dev = params[1]
        else:
            # Default to standard normal
            mean = 0
            std_dev = 1
        
        if parsed['calculation_type'] == 'probability':
            if parsed['inequality'] == '<':
                probability = stats.norm.cdf(parsed['compare_value'], mean, std_dev)
                return {
                    "result": probability,
                    "explanation": f"The probability of X < {parsed['compare_value']} in a normal distribution with mean {mean} and standard deviation {std_dev} is {probability:.4f}"
                }
            elif parsed['inequality'] == '>':
                probability = 1 - stats.norm.cdf(parsed['compare_value'], mean, std_dev)
                return {
                    "result": probability,
                    "explanation": f"The probability of X > {parsed['compare_value']} in a normal distribution with mean {mean} and standard deviation {std_dev} is {probability:.4f}"
                }
            elif parsed['inequality'] == '<=':
                probability = stats.norm.cdf(parsed['compare_value'], mean, std_dev)
                return {
                    "result": probability,
                    "explanation": f"The probability of X ≤ {parsed['compare_value']} in a normal distribution with mean {mean} and standard deviation {std_dev} is {probability:.4f}"
                }
            elif parsed['inequality'] == '>=':
                probability = 1 - stats.norm.cdf(parsed['compare_value'], mean, std_dev)
                return {
                    "result": probability,
                    "explanation": f"The probability of X ≥ {parsed['compare_value']} in a normal distribution with mean {mean} and standard deviation {std_dev} is {probability:.4f}"
                }
            elif parsed['inequality'] == 'between' and parsed['interval']:
                lower, upper = parsed['interval']
                probability = stats.norm.cdf(upper, mean, std_dev) - stats.norm.cdf(lower, mean, std_dev)
                return {
                    "result": probability,
                    "explanation": f"The probability of {lower} < X < {upper} in a normal distribution with mean {mean} and standard deviation {std_dev} is {probability:.4f}"
                }
            else:
                return {"error": "Could not determine the exact probability to calculate."}
        
        elif parsed['calculation_type'] == 'mean':
            return {
                "result": mean,
                "explanation": f"The mean of the normal distribution with parameters μ={mean} and σ={std_dev} is {mean}"
            }
        
        elif parsed['calculation_type'] == 'variance':
            variance = std_dev ** 2
            return {
                "result": variance,
                "explanation": f"The variance of the normal distribution with parameters μ={mean} and σ={std_dev} is {variance}"
            }
        
        elif parsed['calculation_type'] == 'standard_deviation':
            return {
                "result": std_dev,
                "explanation": f"The standard deviation of the normal distribution with parameters μ={mean} and σ={std_dev} is {std_dev}"
            }
        
        elif parsed['calculation_type'] == 'quantile':
            if len(params) >= 3:
                p = params[2]
                if 0 <= p <= 1:
                    quantile = stats.norm.ppf(p, mean, std_dev)
                    return {
                        "result": quantile,
                        "explanation": f"The {p*100}th percentile of the normal distribution with mean {mean} and standard deviation {std_dev} is {quantile:.4f}"
                    }
                else:
                    return {"error": "Percentile must be between 0 and 1"}
            else:
                return {"error": "Not enough parameters to calculate quantile"}
        
        else:
            return {"error": f"Calculation type '{parsed['calculation_type']}' not supported for normal distribution"}
    
    def _solve_binomial(self, parsed):
        """Solve problems related to binomial distribution"""
        params = parsed['parameters']
        
        # Extract n (trials) and p (probability of success)
        if len(params) >= 2:
            n = int(params[0])
            p = params[1]
        else:
            return {"error": "Not enough parameters for binomial distribution. Need number of trials and probability of success."}
        
        if parsed['calculation_type'] == 'probability':
            if parsed['inequality'] == '=' or parsed['inequality'] is None:
                # Exact probability P(X = k)
                if parsed['compare_value'] is not None:
                    k = int(parsed['compare_value'])
                    probability = stats.binom.pmf(k, n, p)
                    return {
                        "result": probability,
                        "explanation": f"The probability of exactly {k} successes in {n} trials with p={p} is {probability:.4f}"
                    }
                else:
                    return {"error": "No value specified for exact probability calculation"}
            
            elif parsed['inequality'] == '<':
                k = int(parsed['compare_value'])
                probability = stats.binom.cdf(k-1, n, p)  # P(X < k) = P(X <= k-1)
                return {
                    "result": probability,
                    "explanation": f"The probability of less than {k} successes in {n} trials with p={p} is {probability:.4f}"
                }
            
            elif parsed['inequality'] == '<=':
                k = int(parsed['compare_value'])
                probability = stats.binom.cdf(k, n, p)
                return {
                    "result": probability,
                    "explanation": f"The probability of at most {k} successes in {n} trials with p={p} is {probability:.4f}"
                }
            
            elif parsed['inequality'] == '>':
                k = int(parsed['compare_value'])
                probability = 1 - stats.binom.cdf(k, n, p)
                return {
                    "result": probability,
                    "explanation": f"The probability of more than {k} successes in {n} trials with p={p} is {probability:.4f}"
                }
            
            elif parsed['inequality'] == '>=':
                k = int(parsed['compare_value'])
                probability = 1 - stats.binom.cdf(k-1, n, p)
                return {
                    "result": probability,
                    "explanation": f"The probability of at least {k} successes in {n} trials with p={p} is {probability:.4f}"
                }
            
            elif parsed['inequality'] == 'between' and parsed['interval']:
                lower, upper = int(parsed['interval'][0]), int(parsed['interval'][1])
                probability = stats.binom.cdf(upper, n, p) - stats.binom.cdf(lower-1, n, p)
                return {
                    "result": probability,
                    "explanation": f"The probability of between {lower} and {upper} successes (inclusive) in {n} trials with p={p} is {probability:.4f}"
                }
            
            else:
                return {"error": "Could not determine the exact probability to calculate."}
        
        elif parsed['calculation_type'] == 'mean':
            mean = n * p
            return {
                "result": mean,
                "explanation": f"The mean of the binomial distribution with n={n} and p={p} is {mean}"
            }
        
        elif parsed['calculation_type'] == 'variance':
            variance = n * p * (1 - p)
            return {
                "result": variance,
                "explanation": f"The variance of the binomial distribution with n={n} and p={p} is {variance}"
            }
        
        elif parsed['calculation_type'] == 'standard_deviation':
            std_dev = np.sqrt(n * p * (1 - p))
            return {
                "result": std_dev,
                "explanation": f"The standard deviation of the binomial distribution with n={n} and p={p} is {std_dev:.4f}"
            }
        
        else:
            return {"error": f"Calculation type '{parsed['calculation_type']}' not supported for binomial distribution"}
    
    def _solve_poisson(self, parsed):
        """Solve problems related to Poisson distribution"""
        params = parsed['parameters']
        
        # Extract lambda (rate parameter)
        if len(params) >= 1:
            lambda_param = params[0]
        else:
            return {"error": "Not enough parameters for Poisson distribution. Need rate parameter (lambda)."}
        
        if parsed['calculation_type'] == 'probability':
            if parsed['inequality'] == '=' or parsed['inequality'] is None:
                # Exact probability P(X = k)
                if parsed['compare_value'] is not None:
                    k = int(parsed['compare_value'])
                    probability = stats.poisson.pmf(k, lambda_param)
                    return {
                        "result": probability,
                        "explanation": f"The probability of exactly {k} events in a Poisson distribution with λ={lambda_param} is {probability:.4f}"
                    }
                else:
                    return {"error": "No value specified for exact probability calculation"}
            
            elif parsed['inequality'] == '<':
                k = int(parsed['compare_value'])
                probability = stats.poisson.cdf(k-1, lambda_param)  # P(X < k) = P(X <= k-1)
                return {
                    "result": probability,
                    "explanation": f"The probability of less than {k} events in a Poisson distribution with λ={lambda_param} is {probability:.4f}"
                }
            
            elif parsed['inequality'] == '<=':
                k = int(parsed['compare_value'])
                probability = stats.poisson.cdf(k, lambda_param)
                return {
                    "result": probability,
                    "explanation": f"The probability of at most {k} events in a Poisson distribution with λ={lambda_param} is {probability:.4f}"
                }
            
            elif parsed['inequality'] == '>':
                k = int(parsed['compare_value'])
                probability = 1 - stats.poisson.cdf(k, lambda_param)
                return {
                    "result": probability,
                    "explanation": f"The probability of more than {k} events in a Poisson distribution with λ={lambda_param} is {probability:.4f}"
                }
            
            elif parsed['inequality'] == '>=':
                k = int(parsed['compare_value'])
                probability = 1 - stats.poisson.cdf(k-1, lambda_param)
                return {
                    "result": probability,
                    "explanation": f"The probability of at least {k} events in a Poisson distribution with λ={lambda_param} is {probability:.4f}"
                }
            
            elif parsed['inequality'] == 'between' and parsed['interval']:
                lower, upper = int(parsed['interval'][0]), int(parsed['interval'][1])
                probability = stats.poisson.cdf(upper, lambda_param) - stats.poisson.cdf(lower-1, lambda_param)
                return {
                    "result": probability,
                    "explanation": f"The probability of between {lower} and {upper} events (inclusive) in a Poisson distribution with λ={lambda_param} is {probability:.4f}"
                }
            
            else:
                return {"error": "Could not determine the exact probability to calculate."}
        
        elif parsed['calculation_type'] == 'mean':
            return {
                "result": lambda_param,
                "explanation": f"The mean of the Poisson distribution with λ={lambda_param} is {lambda_param}"
            }
        
        elif parsed['calculation_type'] == 'variance':
            return {
                "result": lambda_param,
                "explanation": f"The variance of the Poisson distribution with λ={lambda_param} is {lambda_param}"
            }
        
        elif parsed['calculation_type'] == 'standard_deviation':
            std_dev = np.sqrt(lambda_param)
            return {
                "result": std_dev,
                "explanation": f"The standard deviation of the Poisson distribution with λ={lambda_param} is {std_dev:.4f}"
            }
        
        else:
            return {"error": f"Calculation type '{parsed['calculation_type']}' not supported for Poisson distribution"}
    
    def _solve_uniform(self, parsed):
        """Solve problems related to uniform distribution"""
        params = parsed['parameters']
        
        # Extract a (lower bound) and b (upper bound)
        if len(params) >= 2:
            a = params[0]
            b = params[1]
        else:
            return {"error": "Not enough parameters for uniform distribution. Need lower and upper bounds."}
        
        if a >= b:
            return {"error": "Lower bound must be less than upper bound for uniform distribution."}
        
        if parsed['calculation_type'] == 'probability':
            if parsed['inequality'] == '<':
                x = parsed['compare_value']
                if x <= a:
                    probability = 0
                elif x >= b:
                    probability = 1
                else:
                    probability = (x - a) / (b - a)
                return {
                    "result": probability,
                    "explanation": f"The probability of X < {x} in a uniform distribution U({a}, {b}) is {probability:.4f}"
                }
            
            elif parsed['inequality'] == '<=':
                x = parsed['compare_value']
                if x < a:
                    probability = 0
                elif x > b:
                    probability = 1
                else:
                    probability = (x - a) / (b - a)
                return {
                    "result": probability,
                    "explanation": f"The probability of X ≤ {x} in a uniform distribution U({a}, {b}) is {probability:.4f}"
                }
            
            elif parsed['inequality'] == '>':
                x = parsed['compare_value']
                if x <= a:
                    probability = 1
                elif x >= b:
                    probability = 0
                else:
                    probability = (b - x) / (b - a)
                return {
                    "result": probability,
                    "explanation": f"The probability of X > {x} in a uniform distribution U({a}, {b}) is {probability:.4f}"
                }
            
            elif parsed['inequality'] == '>=':
                x = parsed['compare_value']
                if x < a:
                    probability = 1
                elif x > b:
                    probability = 0
                else:
                    probability = (b - x) / (b - a)
                return {
                    "result": probability,
                    "explanation": f"The probability of X ≥ {x} in a uniform distribution U({a}, {b}) is {probability:.4f}"
                }
            
            elif parsed['inequality'] == 'between' and parsed['interval']:
                lower, upper = parsed['interval']
                lower = max(a, lower)
                upper = min(b, upper)
                if lower >= upper:
                    probability = 0
                else:
                    probability = (upper - lower) / (b - a)
                return {
                    "result": probability,
                    "explanation": f"The probability of {lower} < X < {upper} in a uniform distribution U({a}, {b}) is {probability:.4f}"
                }
            
            else:
                return {"error": "Could not determine the exact probability to calculate."}
        
        elif parsed['calculation_type'] == 'mean':
            mean = (a + b) / 2
            return {
                "result": mean,
                "explanation": f"The mean of the uniform distribution U({a}, {b}) is {mean}"
            }
        
        elif parsed['calculation_type'] == 'variance':
            variance = ((b - a) ** 2) / 12
            return {
                "result": variance,
                "explanation": f"The variance of the uniform distribution U({a}, {b}) is {variance}"
            }
        
        elif parsed['calculation_type'] == 'standard_deviation':
            std_dev = (b - a) / np.sqrt(12)
            return {
                "result": std_dev,
                "explanation": f"The standard deviation of the uniform distribution U({a}, {b}) is {std_dev:.4f}"
            }
        
        else:
            return {"error": f"Calculation type '{parsed['calculation_type']}' not supported for uniform distribution"}


@app.route('/api/solve', methods=['POST'])
def solve():
    data = request.json
    if not data or 'question' not in data:
        return jsonify({"error": "No question provided"}), 400
    
    solver = DistributionSolver()
    result = solver.solve(data['question'])
    return jsonify(result)

@app.route('/api/parse', methods=['POST'])
def parse():
    data = request.json
    if not data or 'question' not in data:
        return jsonify({"error": "No question provided"}), 400
    
    solver = DistributionSolver()
    parsed = solver.parse_question(data['question'])
    return jsonify(parsed)

if __name__ == '__main__':
    app.run(debug=True)