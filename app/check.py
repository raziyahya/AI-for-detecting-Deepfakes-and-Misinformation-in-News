import csv
zc=0
oc=0
i=0
listfinal=[]
with open(r'D:\germany\Deep Fake\Deepfake\Training_Essay_Data.csv', mode ='r', encoding='utf-8')as file:
  csvFile = csv.reader(file)
  for lines in csvFile:
      if i!=0:
          try:
              print(lines)
              if lines[1]=="0":
                  zc+=1
                  if zc<1000:
                      print("000000000000000")
                      listfinal.append(lines)
              else:
                  oc+=1
                  if oc<1000:
                      print("1111111111111111")
                      listfinal.append(lines)
          except:
              pass
      else:
          listfinal.append(lines)
      i=i+1
print(zc)
print(oc)
print(i)

file_name = 'example.csv'

# Open the CSV file in write mode
with open(file_name, mode='w', newline='',  encoding='utf-8') as file:
    writer = csv.writer(file)

    # Write rows to the CSV file
    writer.writerows(listfinal)