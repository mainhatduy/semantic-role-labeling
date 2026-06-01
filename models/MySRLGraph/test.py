from src.inference import SRLInferencePipeline

pipeline = SRLInferencePipeline("/teamspace/studios/this_studio/semantic-role-labeling/models/outputs/2026-06-01/12-24-48-srl_edge_diffusion")
result = pipeline.predict("The cat sat on the mat")
print(result)