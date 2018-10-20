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
                ? <Redirect push to={redirect} />
                : <section className="main">
                    <h1 className="main__header">Varian Analyzer</h1>
                    <h3 className="main__description">Here goes some cool description.</h3>
                    <div className="main__buttons">
                        <Button text="Scan" />
                        <Button text="Upload" onClick={ () => this.setRedirect('/predict') } />
                    </div>
                </section>
    }

    render = () => this.renderMain( this.state )

}