from __future__ import absolute_import, print_function, division
import numpy

import theano
import theano.tensor as tt
from six.moves import xrange


theano.config.warn.subtensor_merge_bug = False

k = tt.iscalar("k")
A = tt.vector("A")


def inner_fct(prior_result, A):
    return prior_result * A

result, updates = theano.scan(fn=inner_fct,
                              outputs_info=tt.ones_like(A),
                              non_sequences=A, n_steps=k)

final_result = result[-1]

power = theano.function(inputs=[A, k], outputs=final_result,
                        updates=updates)

print(power(list(range(10)), 2))



coefficients = tt.vector("coefficients")
x = tt.scalar("x")
max_coefficients_supported = 10000

full_range = tt.arange(max_coefficients_supported)
components, updates = theano.scan(fn=lambda coeff, power, free_var:
                                  coeff * (free_var ** power),
                                  sequences=[coefficients, full_range],
                                  outputs_info=None,
                                  non_sequences=x)
polynomial = components.sum()
calculate_polynomial1 = theano.function(inputs=[coefficients, x],
                                        outputs=polynomial)

test_coeff = numpy.asarray([1, 0, 2], dtype=numpy.float32)
print(calculate_polynomial1(test_coeff, 3))


theano.config.warn.subtensor_merge_bug = False

coefficients = tt.vector("coefficients")
x = tt.scalar("x")
max_coefficients_supported = 10000

full_range = tt.arange(max_coefficients_supported)


outputs_info = tt.as_tensor_variable(numpy.asarray(0, 'float64'))

components, updates = theano.scan(fn=lambda coeff, power, prior_value, free_var:
                                  prior_value + (coeff * (free_var ** power)),
                                  sequences=[coefficients, full_range],
                                  outputs_info=outputs_info,
                                  non_sequences=x)

polynomial = components[-1]
calculate_polynomial = theano.function(inputs=[coefficients, x],
                                       outputs=polynomial, updates=updates)

test_coeff = numpy.asarray([1, 0, 2], dtype=numpy.float32)
print(calculate_polynomial(test_coeff, 3))
