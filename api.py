import firebase_admin
from firebase_admin import credentials, firestore
cred = credentials.Certificate("ai-sampling-firebase-adminsdk-hesqz-7052dd026a.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
doc_ref = db.collection(u'test').document(u'hello')
doc_ref.set({ u'first': u'Hello' })

def logRun (
    timestamps: list[float],
    accuracyTrain: list[float],
    accuracyTest: list[float],
    lossTrain: list[float],
    lossTest: list[float],
    networkName: str,
    run: int,
    runName: str,
  ) -> None:
  doc_ref = db.collection(u'networks').document(networkName).collection(u'runs').document(str(run))
  doc_ref.set({
    u'timestamps': timestamps,
    u'accuracyTrain': accuracyTrain,
    u'accuracyTest': accuracyTest,
    u'lossTrain': lossTrain,
    u'lossTest': lossTest,
    u'name': runName,
  }
)

def logNetwork (
    batchSize: int,
    testSize: int,
    name: str,
    lr: float,
    optimizer: str,
    lossFunction: str,
    model: str,
  ) -> None:
  doc_ref = db.collection(u'networks').document(name)
  doc_ref.set({
    u'batchSize': batchSize,
    u'testSize': testSize,
    u'name': name,
    u'lr': lr,
    u'optimizer': optimizer,
    u'lossFunction': lossFunction,
    u'model': model
  })

def log3DNodes (
  nodes,
  networkName: str,
) -> None:
  doc_ref = db.collection(u'networks').document(networkName)
  doc_ref.set({
    u'nodes': nodes,
  }, merge=True)



# logNetwork([1,2,3], [0.1, 0.2, 0.3], 'test')