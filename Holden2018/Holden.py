class Holden():
    def __init__(self):
        return

    def extract_data(self):
        from Holden2018.extract.generate_holden_dataset import generate_holden_dataset
        generate_holden_dataset()
        return

    def train(self):
        from Holden2018.train.train_Holden import train_Holden
        train_Holden()
        return

    def evaluate(self):
        from Holden2018.evaluate.eval_Holden import eval_Holden
        eval_Holden()
        return