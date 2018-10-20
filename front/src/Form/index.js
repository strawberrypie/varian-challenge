import React        from 'react';
import { Redirect } from 'react-router-dom';
import Button       from '../Button';

export default class Form extends React.Component {

    state = {
        redirect: void(0),
        name:     '',
        files:    []
    }

    setRedirect = redirect => this.setState({redirect})

    handleSubmit = (event) => {

        var formData = new FormData();
        // {
        //     meta: {
        //         name: this.state.name
        //     },
        //     data: this.state.files[0]
        // }
        for (let i = 0; i < this.state.files.length; i++) {
            let file = this.state.files[i];

            formData.append('file', file);
        }

        console.log(formData);

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

    renderForm = ({ redirect, name, files }) => {
        return redirect
                ? <Redirect push to={redirect} />
                : <div className="form">
                    <h1 className="form__header">Form</h1>

                    <form onSubmit={ this.handleSubmit }>
                        <label>
                            Identifier:
                            <input
                                type     = "text"
                                name     = "id"
                                value    = { name }
                                onChange = { e => this.setState({ name: e.target.value }) }
                            />
                        </label>
                        <input
                            type     = "file"
                            name     = "data"
                            onChange = { e => this.setState({files: e.target.files }) }
                        />
                        <input type="submit" value="Submit" />
                    </form>

                    <div className="dicomImage"></div>

                    { console.log(this.state) }
                </div>
    }

    render = () => this.renderForm( this.state )

}