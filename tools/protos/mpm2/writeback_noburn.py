import writeback_probe as wp
print("-- no-burn torch (1000 J/tick x 60): the energy ledger alone")
wp.run(False, ticks=80, torch=1000.0, torch_ticks=60)
wp.run(True, ticks=80, torch=1000.0, torch_ticks=60)
