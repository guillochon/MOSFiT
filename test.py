"""Run a test of instantiating `Fitter`, running `fit_events`."""
import mosfit

my_fitter = mosfit.fitter.Fitter()

print('Running `fit_events` test.')

entries, ps, lnprobs = my_fitter.fit_events(
    events=['SN2009do', 'SN2007bg'], models=['magni', 'slsn'], iterations=1,
    quiet=False, test=True, offline=True)

print('Model WAICs: ',
      [[y['models'][0]['score']['value'] for y in x] for x in entries])
