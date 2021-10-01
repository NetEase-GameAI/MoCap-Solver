from Holden2018.Holden import Holden

model_class = Holden()

print('1. Start generating training dataset!')
model_class.extract_data()

print('2. Start training Holden 2018 model!')
model_class.train()

print('3. Start evaluating Holden 2018 model!')
model_class.evaluate()