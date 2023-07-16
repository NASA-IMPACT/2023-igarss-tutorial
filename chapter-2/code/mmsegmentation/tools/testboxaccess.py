from boxsdk import OAuth2, Client

auth = OAuth2(
    client_id='YOUR_CLIENT_ID',
    client_secret='YOUR_CLIENT_SECRET',
    access_token='YOUR_DEVELOPER_TOKEN',
)
client = Client(auth)

user = client.user().get()
print(f'The current user ID is {user.id}')