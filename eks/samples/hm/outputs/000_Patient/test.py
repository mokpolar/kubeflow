from ftplib import FTP


# set FTP

ftp = FTP()
ftp.debug(1)
ftp.connect("ec2-54-204-143-157.compute-1.amazonaws.com", 21)
ftp.login('fftp','kfserving11')
ftp.cwd('./folder')

print('login successful')

# set active mode

ftp.set_pasv(False)
print('active mode')


# upload file
file = '00_slice.png'
ftp.storbinary('STOR '+ file, open(file, "rb"))


print('Upload file is ' + file)
