
st = 0.633691



so = [0.00426978, 0.0116065, 0.0315496, 0.0857608, 0.233122]


s1 = st * (1 - st)

print ("s1:", s1)

sum = 0

for i in range(5):
    sum += st * ( - so[i])

print ("sum:", sum)

print ("s1+sum:", s1 + sum) 