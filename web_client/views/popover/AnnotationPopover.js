import _ from 'underscore';

import View from '../View';
import { restRequest } from 'girder/rest';

import ElementCollection from 'girder_plugins/large_image/collections/ElementCollection';
import annotationPopover from '../../templates/popover/annotationPopover.pug';
import '../../stylesheets/popover/annotationPopover.styl';

/**
 * This view behaves like a bootstrap "popover" that follows the mouse pointer
 * over the image canvas and dynamically updates according to the features
 * under the pointer.
 *
 * @param {object} [settings]
 * @param {number} [settings.debounce]
 *   Debounce time in ms for rerendering due to mouse movements
 */
var AnnotationPopover = View.extend({
    initialize(settings) {
        if (settings.debounce) {
            this._position = _.debounce(this._position, settings.debounce);
        }

        $('body').on('mousemove', '.h-image-view-body', (evt) => this._position(evt));
        $('body').on('mouseout', '.h-image-view-body', () => this._hide());
        $('body').on('mouseover', '.h-image-view-body', () => this._show());

        this._hidden = !settings.visible;
        this._users = {};
        this.collection = new ElementCollection();
        this.listenTo(this.collection, 'add', this._getUser);
        this.listenTo(this.collection, 'all', this.render);
    },

    render() {
        this.$el.html(
            annotationPopover({
                annotations: _.uniq(
                    this.collection.pluck('annotation'),
                    _.property('id')),
                formatDate: this._formatDate,
                users: this._users
            })
        );
        this._show();
        if (!this._visible()) {
            this._hide();
        }
        this._height = this.$('.h-annotation-popover').height();
    },

    /**
     * Set the popover visibility state.
     *
     * @param {boolean} [show]
     *   if true: show the popover
     *   if false: hide the popover
     *   if undefined: toggle the popover state
     */
    toggle(show) {
        if (show === undefined) {
            show = this._hidden;
        }
        this._hidden = !show;
        this.render();
        return this;
    },

    /**
     * Check the local cache for the given creator.  If it has not already been
     * fetched, then send a rest request to get the user information and
     * rerender the popover.
     *
     * As a consequence to avoid always rendering asyncronously, the user name
     * will not be shown on the first render.  In practice, this isn't usually
     * noticeable.
     */
    _getUser(model) {
        var id = model.get('annotation').get('creatorId');
        if (!_.has(this._users, id)) {
            restRequest({
                path: 'user/' + id
            }).then((user) => {
                this._users[id] = user;
                this.render();
            });
        }
    },

    /**
     * Format a Date object as a localized string.
     */
    _formatDate(s) {
        var d = new Date(s);
        return d.toLocaleString();
    },

    /**
     * Remove the hidden class on the popover element if this._visible()
     * returns true.
     */
    _show() {
        if (this._visible()) {
            this.$el.removeClass('hidden');
        }
    },

    /**
     * Unconditionally hide popover.
     */
    _hide() {
        this.$el.addClass('hidden');
    },

    /**
     * Determine if the popover should be visible.  Returns true
     * if there are active annotations under the mouse pointer and
     * the label option is enabled.
     */
    _visible() {
        return !this._hidden && this.collection.length > 0;
    },

    /**
     * Reset the position of the popover to the position of the
     * mouse pointer.
     */
    _position(evt) {
        if (this._visible()) {
            this.$el.css({
                left: evt.pageX + 5,
                top: evt.pageY - this._height / 2
            });
        }
    }
});

export default AnnotationPopover;
