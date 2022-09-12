# need to translate numpy/tf obj into torch

# objective
def objective(x, ke, args, volume_contraint=False, cone_filter=True):
  """Objective function (compliance) for topology optimization."""
  kwargs = dict(penal=args['penal'], e_min=args['young_min'], e_0=args['young'])
  x_phys = physical_density(x, args, volume_contraint=volume_contraint,
                            cone_filter=cone_filter)
  forces = calculate_forces(x_phys, args)
  u = displace(
      x_phys, ke, forces, args['freedofs'], args['fixdofs'], **kwargs)
  c = compliance(x_phys, u, ke, **kwargs)
  return c