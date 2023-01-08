import csv

file_name = 'Chanel2_800nm_0_03'

with open(file_name + '.csv', 'r') as input:
    rows = [row for row in csv.reader(input, delimiter=';')]

with open('Reverse{}.csv'.format(file_name), 'w') as output:
    writer = csv.writer(output, delimiter=';')
    writer.writerows(rows[::-1])
