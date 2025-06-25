from dwdynamics import ComplexDynamicsProblem, Objective, helpers, instance

system = 4
df = helpers.get_velox_results(system)
df = df[df.success_prob >0]
for index, row in df.iterrows():
    precision = int(row.precision)
    timepoints = int(row.timepoints)
    inst = instance.Instance(system)
    inst.create_instance(precision, timepoints)
    inst.
print(df)