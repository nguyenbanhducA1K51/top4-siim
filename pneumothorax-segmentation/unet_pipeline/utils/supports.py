import pandas as pd

x=pd.DataFrame({

    "x":[1,2,3,4],
    "y":[5,6,7,8]


})

x=x[:2]

print (x)
print (len(x))