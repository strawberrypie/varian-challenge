import React        from 'react';
import { Redirect } from 'react-router-dom';
import Button       from '../Button';

export default class Form extends React.Component {

    state = {
        redirect: void(0),
        name:     '',
        files:    void(0),
        loading: false,
        showButton: false
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
        var image;

        event.target.onmouseup = (e) => { e.target.onmousemove = null; return; };
        document.onmouseup     = (e) => { if (image) {image.onmousemove = null;} };

        event.target.onmousemove = (e) => {
            image = e.target;

            e.target.style.webkitFilter = `brightness(${
                50 + (e.clientX - e.target.offsetLeft)/(e.target.offsetWidth / 100)
            }%) contrast(${
                50 + (e.clientY - e.target.offsetTop)/(e.target.offsetHeight / 100)
            }%)`;

            !this.state.showButton && this.setState({ showButton: true })
        }
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

    renderResults = (res, currentImage, showButton) =>
        <div className="form__results">
            <div className="form__results-list">
                {
                    res.map(
                        ({image, percent}, index) =>
                            <div
                                key       = {`img-${index}`}
                                tabIndex  = {index + 1}
                                className = {
                                    'form__result' +
                                    ( (percent > 0.75) ? ' form__result-critical' : '' )
                                }
                                onClick   = {() => this.setState({ currentImage: {image, percent} })}
                                onFocus   = {(e) => {
                                    document.querySelector('.form__image img').style.webkitFilter = '';
                                    this.setState({ currentImage: {image, percent}, showButton: false });
                                }}
                            >
                                {`Image ${index}`}
                            </div>
                    )
                }
            </div>
            <div className="form__image">
                <div>{`Certainty: ${currentImage.percent.toFixed(2) * 100}%`}</div>
                <img
                    src       = {`data:image/png;base64,${currentImage.image}`}
                    draggable = {false}
                    onMouseDown = {(e) => this.handleMouseMove(e)}
                />
                {
                    showButton &&
                    <Button
                        text="Reset filter"
                        onClick={() => {
                            document.querySelector('.form__image img').style.webkitFilter = '';
                            this.setState({ showButton: false });
                        }}
                    />
                }
            </div>
        </div>

    renderPredict = ({ redirect, files, label, res, loading, currentImage, showButton }) => {
        return redirect
                ? <Redirect push to={redirect} />
                : <section className="form">
                    <h1 className="form__header">Prediction</h1>
                    {
                        res
                            ? this.renderResults(res, currentImage || res[0], showButton)
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