import React, { Component } from 'react';
import { Route, Switch }  from 'react-router-dom';
import Main from './Main';
import Form from './Form';
import Result from './Result';
import './App.css';

class App extends Component {
    render() {
        return (
            <div className="App">
                <Switch>
                    <Route exact path="/"        component={ Main   } />
                    <Route       path="/predict" component={ Form   } />
                    <Route       path="/result"  component={ Result } />
                </Switch>
            </div>
        );
    }
}

export default App;
