import firebase_admin
from firebase_admin import credentials, firestore
cred = credentials.Certificate("ai-sampling-firebase-adminsdk-hesqz-7052dd026a.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
doc_ref = db.collection(u'test').document(u'hello')
doc_ref.set({ u'first': u'Hello' })


def logNetwork (timestamps: list, accuracyTraining, accuracyTest, loss, batchSize, testSize, name, lr, optimizer, lossFunction, model):
  doc_ref = db.collection(u'networks').document(name)
  doc_ref.set({
    u'timestamps': timestamps,
    u'accuracyTraining': accuracyTraining,
    u'accuracyTest': accuracyTest,
    u'loss': loss,
    u'batchSize': batchSize,
    u'testSize': testSize,
    u'name': name,
    u'lr': lr,
    u'optimizer': optimizer,
    u'lossFunction': lossFunction,
    u'model': model
  })


# logNetwork([1,2,3], [0.1, 0.2, 0.3], 'test')