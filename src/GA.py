import random
from deap import base, creator, tools
from datasets import load_metric

class GaTextSummarizetionOptimizer:
    def __init__(self) -> None:
        self.metric = load_metric('rouge',trust_remote_code=True)
        self.toolbox = base.Toolbox()
        GaTextSummarizetionOptimizer.initialize_ga()
        GaTextSummarizetionOptimizer.configure_toolbox()

    def run():
        pass

    @classmethod
    def initialize_ga(cls):
        creator.create("FitnessMax",base=base.Fitness,weights=(1.0,))
        creator.create("Individual",float,fitness=creator.FitnessMax)


    def evaluation(self,Individual,target)->tuple:
        """
        calucalate ROUGE-N between Individual and target 
        summary
        """
        
        ref = [Individual]
        generated = [target]
        return self.metric.compute(predictions=generated,references=ref)['rougeL'][-1][0],

    @classmethod
    def deletion_mutate(cls,individual,ratio)->tuple:
        if random.random() < ratio:
            return 0.0,
        return individual,

    def configure_toolbox(self,
                            Population_size,
                            target,
                            mutation_ratio=0.01,
                        ):

        # register the initialization of each chromosome
        self.toolbox.register("attribute",random.uniform,0,1)
        # register each chromosome and number of generated ones
        self.toolbox.register("individual",
                        tools.initRepeat,
                        creator.Individual,
                        self.toolbox.attribute,
                        n=Population_size)
        # register population as list of chromosomes
        self.toolbox.register("population",
                        tools.initRepeat,
                        list,
                        self.toolbox.individual)
        # register the evaluation function (fitness fn)
        self.toolbox.register("evaluation",GaTextSummarizetionOptimizer.evaluation,target)
        # register the type of mutation
        self.toolbox.register("mate",tools.cxTwoPoint,)
        # register the mutation ration
        self.toolbox.register("mutate",GaTextSummarizetionOptimizer.deletion_mutate,mutation_ratio)
        # register the selection technique
        self.toolbox.register("select",tools.selTournament,toursize=5)


if __name__ == '__main__':
    pass
