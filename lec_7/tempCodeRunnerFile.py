prompt = "Python is "
next_token_probabilities = { 
"useful": 0.05,
"powerful": 0.25,
"popular": 0.1500,
"creative": 0.10
          }
best_token = max(next_token_probabilities,key=next_token_probabilities.get )
print(prompt+best_token)