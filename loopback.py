import sys

args = sys.argv[1:]

try:
    mcname = args[0]
    modname = args[1]
    oname = args[2]
except:
    print("Usage: loopback.py [mcmc_input] [modparams] [new mcmc_input]")
    exit()


newvalues = {}
with open(modname, 'r') as f:
    for line in f:
        line = line.strip()
        if line == '':
            break

        line = line.split(' ')

        par = line[0]
        value = float(line[2])

        newvalues[par] = value

pars = newvalues.keys()

mcmc_file = []
with open(mcname, 'r') as f:
    for line in f:
        mcmc_file.append(line)
        line = line.strip()

        line_components = line.split()
        if len(line_components) > 0:
            par = line_components[0]
            if par in pars:
                value = newvalues[par]
                # print("\n\nI know this one! line:\n{}\nPar:  {}\nValue:  {}".format(line, par, value))
                newline = line_components.copy()
                newline[2] = value
                newline = "{:>10} = {:>12} {:>12} {:>12} {:>12} {:>12}\n".format(
                    newline[0], 
                    newline[2], 
                    newline[3], 
                    newline[4], 
                    newline[5], 
                    newline[6]
                    )
                # print("Altered line:")
                # print(newline)
                mcmc_file[-1] = newline

if oname[-4:] != '.dat':
    oname += '.dat'
print('Writing new file, {}'.format(oname))
with open(oname, 'w') as f:
    for line in mcmc_file:
        f.write(line)
