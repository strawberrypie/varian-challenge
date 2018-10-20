import { BehaviorSubject }  from 'rxjs/BehaviorSubject';

var DataStore = new BehaviorSubject({ loading: false, data: void(0) });

DataStore.mixState = ($) =>
    DataStore.next({
     ...DataStore.getValue(),
     ...$
    });

export { DataStore };