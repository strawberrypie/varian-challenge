import React        from 'react';
import { Redirect } from 'react-router-dom';

export default class Form extends React.Component {

    state = {
        redirect: void(0),
        name:     '',
        files:    void(0),
        loading: false
    }

    setRedirect = redirect => this.setState({redirect})

    handleSubmit = (event) => {

        var formData = new FormData();

        for (let i = 0; i < this.state.files.length; i++) {
            let file = this.state.files[i];

            formData.append('file', file);
        }

        this.setState({loading: true});

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
        .then( res => this.setState({res, loading: false}) );

        event.preventDefault();
    }

    handleMouseMove = event => {
        // console.log(
        //     'down',
        //     event,
        //     event.clientX,
        //     event.clientY,
            // event.clientX - event.target.offsetLeft,
        //     event.clientY - event.target.offsetTop
        // );
        document.onmousemove( (e) => console.log( 'move', e.clientX - event.target.offsetLeft, e.clientY - event.target.offsetTop ) );
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

    renderResults = (res, currentImage) =>
        <div className="form__results">
            <div className="form__results-list">
                {
                    res.map(
                        (img, index) =>
                            <div
                                key       = {`img-${index}`}
                                tabIndex  = {index + 1}
                                className = "form__result"
                                onClick   = {() => this.setState({ currentImage: img })}
                                onFocus   = {() => this.setState({ currentImage: img })}
                            >
                                {`Image ${index}`}
                            </div>
                    )
                }
            </div>
            <div className="form__image">
                <img
                    src       = {`data:image/png;base64,${currentImage}`}
                    draggable = {false}
                    // onMouseDown = {(e) => this.handleMouseMove(e)}
                    // onMouseMove   = {(e) => this.setFilter(e)}
                />
                {/*<label htmlFor="brightness">
                    brightness
                    <input type="range" className="form__image-brightness" id="brightness"/>
                </label>
                <label htmlFor="contrast">
                    contrast
                    <input type="range" className="form__image-contrast" id="contrast"/>
                </label>*/}
            </div>
        </div>

    renderPredict = ({ redirect, files, label, res, loading, currentImage }) => {
        return redirect
                ? <Redirect push to={redirect} />
                : <section className="form">
                    <h1 className="form__header">Prediction</h1>
                    {
                        res
                            ? this.renderResults(res, currentImage || res[0])
                            : loading
                                ? <div className="form__loading">
                                    <div className="form__loading-text">loading</div>
                                    <svg width="100" height="30">
                                        <circle fill="#ffffff" id="cLeft"   cx="20" cy="15" r="10" />
                                        <circle fill="#ffffff" id="cCentre" cx="50" cy="15" r="10" />
                                        <circle fill="#139eca" id="cRight"  cx="80" cy="15" r="10" />
                                    </svg>
                                </div>
                                : this.renderForm(label, files)
                    }


                    { console.log(this.state) }
                </section>
    }

    render = () => this.renderPredict( this.state )

}