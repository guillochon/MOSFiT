import mosfit

my_fitter = mosfit.fitter.Fitter()

entries, ps, lnprobs = my_fitter.fit_events(
    events=['LSQ12dlf', 'SN2007bg'], models=['magni', 'slsn'], iterations=1, quiet=True)

print('Model WAICs: ',
      [[y['models'][0]['score']['value'] for y in x] for x in entries])
