import React        from 'react';
import { Redirect } from 'react-router-dom';

export default class Form extends React.Component {

    state = {
        redirect: void(0),
        name:     '',
        files:    void(0)
    }

    setRedirect = redirect => this.setState({redirect})

    handleSubmit = (event) => {

        var formData = new FormData();

        for (let i = 0; i < this.state.files.length; i++) {
            let file = this.state.files[i];

            formData.append('file', file);
        }

        fetch(
            'http://127.0.0.1:5000/v1/predict',
            {
                method: 'POST',
                body: formData
            }
        ).then(response => {
            console.log(response);
        });

        event.preventDefault();
    }

    renderForm = ({ redirect, files, label }) => {
        return redirect
                ? <Redirect push to={redirect} />
                : <section className="form">
                    <h1 className="form__header">Prediction</h1>

                    <form className="form__form" onSubmit={ this.handleSubmit }>
                        <input
                            type      = "file"
                            name      = "data"
                            id        = "data"
                            className = "form__inputfile"
                            onChange  = {
                                e => this.setState({
                                        files: e.target.files,
                                        label: e.target.value.split('\\').pop()
                                    })
                            }
                        />
                        <label htmlFor = "data">
                            {
                                label ||
                                <span className="button">Select .zip file...</span>
                            }
                        </label>

                        {
                            files &&
                            <input
                                type      = "submit"
                                value     = "Submit"
                                className = "button"
                            />
                        }
                    </form>

                    <div className="form__image"></div>

                    { console.log(this.state) }
                </section>
    }

    render = () => this.renderForm( this.state )

}