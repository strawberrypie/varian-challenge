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
                    <h1 className="main__header">
                        <span>BIAS/</span>
                        <span className="main__header-emph">VARIAN</span>
                        <span>CE</span>
                    </h1>
                    <h2 className="main__description">Predict early, live long.</h2>
                    <div className="main__buttons">
                        <Button text="Upload" onClick={ () => this.setRedirect('/predict') } />
                    </div>
                </section>
    }

    render = () => this.renderMain( this.state )

}