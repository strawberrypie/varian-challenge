import React        from 'react';
import { Redirect } from 'react-router-dom';
import Button       from '../Button';

export default class Main extends React.Component {

    state = {
        redirect: void(0)
    }

    setRedirect = (redirect) => this.setState({redirect})

    renderMain = ({redirect}) => {
        return redirect
                ? <Redirect to={redirect} />
                : <div className="main">
                    <h1>Varian Analyzer</h1>
                    <Button text="Scan" />
                    <Button text="Upload" onClick={ () => this.setRedirect('/predict') } />
                </div>
    }

    render = () => this.renderMain( this.state )

}