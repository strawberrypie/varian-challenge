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

        var formData = {
            meta: {
                name: this.state.name
            },
            data: this.state.files[0]
        }

        console.log(formData);

        fetch(
            'http://127.0.0.1:5000/',
            {
                method: 'POST',
                body: formData,
                headers: {
                    'Content-Type': 'application/json; charset=utf-8'
                }
            }
        ).then(response => {
            console.log(response);
        });

        event.preventDefault();
    }

    renderForm = ({ redirect, name, files }) => {
        return redirect
                ? <Redirect to={redirect} />
                : <div className="form">
                    <h1>Form</h1>

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