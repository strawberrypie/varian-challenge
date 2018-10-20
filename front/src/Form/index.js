import React        from 'react';
import { Redirect } from 'react-router-dom';

export default class Form extends React.Component {

    state = {
        redirect: void(0),
        name:     '',
        files:    void(0)
    }

    setRedirect = redirect => this.setState({redirect})

    drawImage = res => {
        var image = new Image();
        image.src = res[0];
        document.querySelector('.form__image').appendChild(image);
    }

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
        )
        .then(
            res => {
                try {
                   return res.json();
                } catch (err) {
                   throw `FETCH failed: ${res.status} ${res.statusText} ${err}`;
                }
            }
        )
        .then( res => this.setState({res}) );

        event.preventDefault();
    }

    renderForm = (label, files) =>
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

    renderResults = res =>
        <div>
            <div className="form__results"></div>
            <div className="form__image">
                {
                    res &&
                    <img src={res[0]}/>
                }
            </div>
        </div>

    renderPredict = ({ redirect, files, label, res }) => {
        return redirect
                ? <Redirect push to={redirect} />
                : <section className="form">
                    <h1 className="form__header">Prediction</h1>
                    {
                        res
                            ? <div className="form__image">
                                {
                                    res &&
                                    <img src={res[0]}/>
                                }
                            </div>
                            : this.renderForm(label, files)
                    }


                    { console.log(this.state) }
                </section>
    }

    render = () => this.renderPredict( this.state )

}