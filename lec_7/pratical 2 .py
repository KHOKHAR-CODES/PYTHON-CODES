prompt = "Python is "
next_token_probabilities = { 
"useful": 0.05,
"powerful": 0.05,
"popular": 0.1,
"creative": 0.00
          }
best_token = min(next_token_probabilities,key=next_token_probabilities.get )
print(prompt+best_token)