"""Run a test of instantiating `Fitter`, running `fit_events`."""
import mosfit
import numpy as np

# Test running the fitter.
my_fitter = mosfit.fitter.Fitter(quiet=False, test=True, offline=True)

print('Running `fit_events` test.')
entries, ps, lnprobs = my_fitter.fit_events(
    events=['SN2009do', 'SN2007bg'], models=['magni', 'slsn'], iterations=1,
    user_fixed_parameters=['covariance'])

print('Model WAICs: ',
      [[y['models'][0]['score']['value'] for y in x] for x in entries])

# Test a single call to the model.
print('Testing single call to Model.likelihood().')
my_fetcher = mosfit.fetcher.Fetcher()

fetched = my_fetcher.fetch('SN2009do')[0]

my_model = mosfit.model.Model(model='slsn')

my_model.load_data(my_fetcher.load_data(fetched), event_name=fetched['name'])

x = np.random.rand(my_model.get_num_free_parameters())
likelihood = my_model.likelihood(x)

print('Model likelihood: `{}`'.format(likelihood))

outputs = my_model.run(x)

print('Keys in output: `{}`'.format(', '.join(list(outputs.keys()))))
